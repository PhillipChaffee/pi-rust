//! The package manager, upstream's `src/core/package-manager.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements this module records:
//!
//! - **The npm source kind drops** (ADR 0007). The dual-channel grammar is
//!   `crate:<name>@<version>` (compile-on-install with `--locked`),
//!   `github:` and the other git URL spellings (pinned-ref checkouts), a
//!   tarball URL, and a local path. An `npm:` source is a parse error, and
//!   every npm-machine restatement upstream carries (the command argv
//!   builders, the global-root lookups, the legacy pnpm paths, the
//!   range-satisfying update checks, and the git-checkout dependency
//!   installs) drops with it.
//! - **Installer safety** (ADR 0007): every fetched channel stages into the
//!   install root and moves into place with one atomic rename; the tarball
//!   channel caps the download and unpack size, rejects traversal and
//!   escaping symlinks; fetched installs write a provenance marker with the
//!   per-file sha256 hash receipt the extension host re-verifies at every
//!   spawn, failing closed (the receipt file is
//!   `pi-package-install.json`, format below in [`InstallReceipt`]).
//! - **The extension entry filter restates to executability**: upstream's
//!   `\.(ts|js)$` file pattern selects TS modules jiti loads; the
//!   Rust-native mechanism spawns binaries, so an extension entry is a file
//!   with an execute bit, the `index.ts`/`index.js` convention restates to
//!   an executable `index`, and the `crate:` channel's cargo-installed
//!   binaries surface through a `bin/` convention directory.
//! - **The command spawner is a seam** ([`CommandRunner`]): upstream's tests
//!   `vi.spyOn` the private `runCommand*` methods; the port injects a
//!   recording double instead.
//! - Upstream's `/proc/self/environ` re-read guards a node `process.env`
//!   cleared by `env -i`; Rust commands inherit the real environment and
//!   the case has no counterpart.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokio_util::sync::CancellationToken;

use pi_ai::http::HttpClient;

use crate::config::{CONFIG_DIR_NAME, EnvLookup};
use crate::settings_manager::{SettingsManager, SettingsStorage};
use crate::utils::git::{GitSource, parse_git_url};
use crate::utils::management_http::{FetchRetryOptions, fetch_with_retry};
use crate::utils::pi_user_agent::get_pi_user_agent;
use sha2::Digest;

/// The network budget the git and registry checks share, upstream's
/// `NETWORK_TIMEOUT_MS`.
pub const NETWORK_TIMEOUT_MS: u64 = 10_000;
/// The parallel version-check workers, upstream's `UPDATE_CHECK_CONCURRENCY`.
pub const UPDATE_CHECK_CONCURRENCY: usize = 4;
/// The parallel git-update workers, upstream's `GIT_UPDATE_CONCURRENCY`.
pub const GIT_UPDATE_CONCURRENCY: usize = 4;

/// The tarball channel's download cap.
pub const MAX_TARBALL_BYTES: u64 = 256 * 1024 * 1024;
/// The tarball channel's total unpack cap.
pub const MAX_UNPACKED_BYTES: u64 = 1024 * 1024 * 1024;
/// The tarball channel's entry-count cap.
pub const MAX_TARBALL_ENTRIES: u64 = 10_000;

/// Whether offline mode is enabled, upstream's `isOfflineModeEnabled`:
/// `PI_OFFLINE` counts when `1`, `true`, or `yes` (case-insensitive) — a
/// stricter vocabulary than the version-check's truthiness.
pub(crate) fn is_offline_mode_enabled_with(env: &EnvLookup) -> bool {
    let Some(value) = env("PI_OFFLINE") else {
        return false;
    };
    value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
}

/// One resolved resource path with its provenance, upstream's
/// `PathMetadata`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathMetadata {
    /// The settings source the resource came from, upstream's `source`
    /// (`"local"`, `"auto"`, or the package source string).
    pub source: String,
    /// Which scope contributed it, upstream's `scope`.
    pub scope: SourceScope,
    /// Whether a package carried it or the agent dirs discovered it,
    /// upstream's `origin`.
    pub origin: ResourceOrigin,
    /// The directory settings-relative entries resolve against, upstream's
    /// `baseDir?`.
    pub base_dir: Option<String>,
}

/// Where a resource's provenance claims it came from, upstream's `origin`
/// union.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceOrigin {
    /// A package delivered it, upstream's `"package"`.
    Package,
    /// A settings array or auto-discovery contributed it, upstream's
    /// `"top-level"`.
    TopLevel,
}

impl ResourceOrigin {
    /// The wire's tag, upstream's union member.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Package => "package",
            Self::TopLevel => "top-level",
        }
    }
}

/// One resolved resource, upstream's `ResolvedResource`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedResource {
    /// The absolute path, upstream's `path`.
    pub path: String,
    /// Whether the resource loads, upstream's `enabled`.
    pub enabled: bool,
    /// The provenance, upstream's `metadata`.
    pub metadata: PathMetadata,
}

/// The four resolved resource lists, upstream's `ResolvedPaths`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedPaths {
    /// Extension entries, upstream's `extensions`.
    pub extensions: Vec<ResolvedResource>,
    /// Skill entries, upstream's `skills`.
    pub skills: Vec<ResolvedResource>,
    /// Prompt template entries, upstream's `prompts`.
    pub prompts: Vec<ResolvedResource>,
    /// Theme entries, upstream's `themes`.
    pub themes: Vec<ResolvedResource>,
}

/// What `resolve` does with a missing package source, upstream's
/// `MissingSourceAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingSourceAction {
    /// Install it, upstream's `"install"`.
    Install,
    /// Continue without it, upstream's `"skip"`.
    Skip,
    /// Fail the resolve, upstream's `"error"`.
    Error,
}

/// A progress event, upstream's `ProgressEvent`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressEvent {
    /// The lifecycle phase, upstream's `type`.
    pub event_type: ProgressEventType,
    /// What the manager is doing, upstream's `action`.
    pub action: ProgressAction,
    /// The source the action runs against, upstream's `source`.
    pub source: String,
    /// The human message, upstream's `message?`.
    pub message: Option<String>,
}

/// The lifecycle phases, upstream's `type` union.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEventType {
    /// An operation began, upstream's `"start"`.
    Start,
    /// Mid-operation detail, upstream's `"progress"`.
    Progress,
    /// An operation finished, upstream's `"complete"`.
    Complete,
    /// An operation failed, upstream's `"error"`.
    Error,
}

/// The operations, upstream's `action` union.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressAction {
    /// `pi install`, upstream's `"install"`.
    Install,
    /// `pi remove`, upstream's `"remove"`.
    Remove,
    /// `pi update`, upstream's `"update"`.
    Update,
    /// A git clone, upstream's `"clone"`.
    Clone,
    /// A temporary git refresh, upstream's `"pull"`.
    Pull,
}

/// The progress callback, upstream's `ProgressCallback`.
pub type ProgressCallback = Box<dyn Fn(&ProgressEvent) + Send + Sync>;

/// An available update, upstream's `PackageUpdate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageUpdate {
    /// The configured source string, upstream's `source`.
    pub source: String,
    /// The display name, upstream's `displayName`.
    pub display_name: String,
    /// The channel, upstream's `type` (`"npm"` restates to `"crate"`).
    pub update_type: PackageUpdateType,
    /// The scope the entry is configured at, upstream's `scope`.
    pub scope: SourceScope,
}

/// The update channels, upstream's `type` union minus the dropped npm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageUpdateType {
    /// A `crate:` source, upstream's `"npm"` restated.
    Crate,
    /// A git source, upstream's `"git"`.
    Git,
}

impl PackageUpdateType {
    /// The wire's tag, upstream's union member.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Crate => "crate",
            Self::Git => "git",
        }
    }
}

/// One configured package entry, upstream's `ConfiguredPackage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfiguredPackage {
    /// The source string, upstream's `source`.
    pub source: String,
    /// Which settings list holds it, upstream's `scope`.
    pub scope: SourceScope,
    /// Whether an object entry filters its resources, upstream's `filtered`.
    pub filtered: bool,
    /// The install path when it exists, upstream's `installedPath?`.
    pub installed_path: Option<String>,
}

/// The scopes a source resolves at, upstream's `SourceScope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceScope {
    /// User settings, upstream's `"user"`.
    User,
    /// Project settings, upstream's `"project"`.
    Project,
    /// A transient `resolveExtensionSources` call, upstream's `"temporary"`.
    Temporary,
}

/// The installable scopes, upstream's `InstalledSourceScope`.
pub type InstalledSourceScope = SourceScope;

/// A parsed source, upstream's `ParsedSource` over the dual-channel grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedSource {
    /// A `crate:` source, upstream's `NpmSource` restated.
    Crate(CrateSource),
    /// A git source, upstream's `GitSource` (the landed [`GitSource`]).
    Git(GitSource),
    /// A tarball URL, the prebuilt channel's explicit spelling.
    Tarball(TarballSource),
    /// A local path, upstream's `LocalSource`.
    Local(LocalSource),
}

/// A `crate:` source, upstream's `NpmSource` fields restated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateSource {
    /// The crate name, upstream's `name`.
    pub name: String,
    /// The exact version when the spec carries one, upstream's `version?`.
    pub version: Option<String>,
    /// Whether the version pins the install, upstream's `pinned` — an
    /// exact semver version pins; anything else floats.
    pub pinned: bool,
}

/// A tarball source: an `http(s)` URL whose path ends in `.tgz` or
/// `.tar.gz`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TarballSource {
    /// The URL, upstream has no counterpart — the npm tarball spelling
    /// carried the URL inside the spec.
    pub url: String,
}

/// A local source, upstream's `LocalSource`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSource {
    /// The path as written, upstream's `path`.
    pub path: String,
}

/// The manager failure, upstream's thrown `Error`s restated as one type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageManagerError(pub String);

impl std::fmt::Display for PackageManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PackageManagerError {}

impl From<String> for PackageManagerError {
    fn from(message: String) -> Self {
        Self(message)
    }
}

/// The resource types, upstream's `ResourceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// Extensions, upstream's `"extensions"`.
    Extensions,
    /// Skills, upstream's `"skills"`.
    Skills,
    /// Prompt templates, upstream's `"prompts"`.
    Prompts,
    /// Themes, upstream's `"themes"`.
    Themes,
}

impl ResourceType {
    /// The directory and settings-field name, upstream's union member.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Extensions => "extensions",
            Self::Skills => "skills",
            Self::Prompts => "prompts",
            Self::Themes => "themes",
        }
    }

    /// The value a parsed settings entry carries for the type.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [Self::Extensions, Self::Skills, Self::Prompts, Self::Themes]
    }
}

/// The per-type entry filter, upstream's `FILE_PATTERNS`. The extensions
/// pattern restates from `\.(ts|js)$` to the executable-bit check.
fn file_matches(resource_type: ResourceType, path: &Path) -> bool {
    match resource_type {
        ResourceType::Extensions => is_executable(path),
        ResourceType::Skills | ResourceType::Prompts => {
            path.extension().is_some_and(|extension| extension == "md")
        }
        ResourceType::Themes => path
            .extension()
            .is_some_and(|extension| extension == "json"),
    }
}

/// Whether the file carries any execute bit, the restated extension filter.
pub(crate) fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).is_ok_and(|meta| meta.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        std::fs::metadata(path).is_ok_and(|meta| !meta.permissions().readonly())
    }
}

/// The ignore-file names the discovery walks, upstream's `IGNORE_FILE_NAMES`.
const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

/// The npm-`ignore` matcher, restated on the Rust `ignore` crate's gitignore
/// matcher — the same substitution the skills loader made (the map's
/// agent-core ticket): case-insensitive by default, prefix-prefixed rules
/// accumulate into one matcher for the walk.
struct IgnoreMatcher(ignore::gitignore::Gitignore);

impl IgnoreMatcher {
    /// Whether a walk path is ignored, npm-`ignore`'s `ignores()` semantics
    /// restated: a directory rule ignores everything under it, so the walk
    /// tests every ancestor prefix as a directory before the path itself.
    fn ignores(&self, path: &str, is_dir: bool) -> bool {
        let mut prefix = String::new();
        for segment in path.split('/') {
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(segment);
            if prefix == path {
                break;
            }
            if self.0.matched(Path::new(&prefix), true).is_ignore() {
                return true;
            }
        }
        self.0.matched(Path::new(path), is_dir).is_ignore()
    }
}

fn to_posix_path(path: &str) -> String {
    path.replace(std::path::MAIN_SEPARATOR, "/")
}

fn get_home_dir(env: &EnvLookup) -> String {
    env("HOME")
        .filter(|home| !home.is_empty())
        .unwrap_or_else(crate::config::home_dir)
}

/// The managed temporary extension folder, upstream's
/// `getExtensionTempFolder`: `<agentDir>/tmp/extensions`, created with
/// owner-only permissions and re-chmodded in case a previous run widened it.
///
/// # Errors
/// A filesystem failure creating or chmodding the directory.
pub fn get_extension_temp_folder(agent_dir: &Path) -> std::io::Result<PathBuf> {
    let temp_folder = agent_dir.join("tmp").join("extensions");
    std::fs::create_dir_all(&temp_folder)?;
    set_owner_only_permissions(&temp_folder);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&temp_folder, std::fs::Permissions::from_mode(0o700));
    }
    Ok(temp_folder)
}

fn set_owner_only_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

/// Prefix one ignore-file line with its directory's path, upstream's
/// `prefixIgnorePattern`: comments drop, `!` negation survives the prefix,
/// an escaped `\!` keeps its literal bang, a leading `/` (root-relative)
/// strips before the prefix joins.
fn prefix_ignore_pattern(line: &str, prefix: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('#') && !trimmed.starts_with("\\#") {
        return None;
    }

    let mut pattern = line.to_string();
    let mut negated = false;
    if pattern.starts_with('!') {
        negated = true;
        pattern = pattern[1..].to_string();
    } else if let Some(rest) = pattern.strip_prefix("\\!") {
        pattern = rest.to_string();
    }
    let pattern = pattern.strip_prefix('/').unwrap_or(&pattern).to_string();
    let prefixed = if prefix.is_empty() {
        pattern
    } else {
        format!("{prefix}{pattern}")
    };
    Some(if negated {
        format!("!{prefixed}")
    } else {
        prefixed
    })
}

fn add_ignore_rules(
    matcher: &mut ignore::gitignore::GitignoreBuilder,
    dir: &Path,
    root_dir: &Path,
) {
    let Some(relative_dir) = root_dir_strip(root_dir, dir) else {
        return;
    };
    let prefix = if relative_dir.as_os_str().is_empty() {
        String::new()
    } else {
        format!("{}/", to_posix_path(&relative_dir.to_string_lossy()))
    };

    for filename in IGNORE_FILE_NAMES {
        let ignore_path = dir.join(filename);
        if !ignore_path.exists() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&ignore_path) else {
            continue;
        };
        for line in content.lines() {
            let Some(pattern) = prefix_ignore_pattern(line, &prefix) else {
                continue;
            };
            let _ = matcher.add_line(None, &pattern);
        }
    }
}

/// The relative path from the walk root to the current directory, upstream's
/// `relative(rootDir, dir)`; `None` when `dir` is not under the root.
fn root_dir_strip(root_dir: &Path, dir: &Path) -> Option<PathBuf> {
    dir.strip_prefix(root_dir).map(Path::to_path_buf).ok()
}

fn is_pattern(entry: &str) -> bool {
    entry.starts_with('!')
        || entry.starts_with('+')
        || entry.starts_with('-')
        || entry.contains('*')
        || entry.contains('?')
}

fn is_override_pattern(entry: &str) -> bool {
    entry.starts_with('!') || entry.starts_with('+') || entry.starts_with('-')
}

fn has_glob_pattern(entry: &str) -> bool {
    entry.contains('*') || entry.contains('?')
}

/// Glob a manifest entry, upstream's `expandPackageGlob`: matches resolve
/// under the root, dot-segment paths drop, and the results sort lexically —
/// the sort pins the manifest's glob-expansion order so the "first wins"
/// collision rule sees a stable sequence.
fn expand_package_glob(pattern: &str, root: &Path) -> Vec<String> {
    let mut matches: Vec<PathBuf> = glob::glob(&build_glob_pattern(pattern, root))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|path| {
            path.strip_prefix(root).is_ok_and(|relative| {
                relative.components().all(|component| match component {
                    std::path::Component::Normal(part) => !part.to_string_lossy().starts_with('.'),
                    _ => true,
                })
            })
        })
        .collect();
    matches.sort();
    matches
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

/// Join a manifest-relative pattern with its root, upstream's `globSync`
/// `cwd` option. The Rust `glob` crate matches whole strings, so the
/// pattern prefixes with the root and normalizes the separators.
fn build_glob_pattern(pattern: &str, root: &Path) -> String {
    let normalized = pattern.replace('\\', "/");
    let root_str = root
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/");
    let joined = normalized.strip_prefix("./").map_or_else(
        || {
            if normalized.starts_with('/') {
                format!("{root_str}{normalized}")
            } else {
                format!("{root_str}/{normalized}")
            }
        },
        |rest| format!("{root_str}/{rest}"),
    );
    joined.replace("//", "/")
}

fn split_patterns(entries: &[String]) -> (Vec<String>, Vec<String>) {
    let mut plain = Vec::new();
    let mut patterns = Vec::new();
    for entry in entries {
        if is_pattern(entry) {
            patterns.push(entry.clone());
        } else {
            plain.push(entry.clone());
        }
    }
    (plain, patterns)
}

/// The matcher the walks share: prefix-prefixed rules from `dir`'s ignore
/// files, rooted at the walk root. `add_ignore_rules` swallows the
/// per-line parse failures, so only globs that validated reach the builder
/// and `build` cannot fail here.
#[expect(
    clippy::expect_used,
    reason = "the builder only registers globs add_line already validated; a build failure is a library invariant break, not a runtime condition"
)]
fn build_ignore_matcher(dir: &Path, root: &Path) -> IgnoreMatcher {
    let mut builder = ignore::gitignore::GitignoreBuilder::new(root.to_string_lossy().into_owned());
    add_ignore_rules(&mut builder, dir, root);
    IgnoreMatcher(builder.build().expect("gitignore builder output builds"))
}

/// Walk a directory collecting files matching the resource's filter,
/// upstream's `collectFiles`: dot entries and `node_modules` skip, symlink
/// targets stat through, ignore files read per directory against the root,
/// errors swallow.
fn collect_files(
    dir: &Path,
    resource_type: ResourceType,
    skip_node_modules: bool,
    matcher: &IgnoreMatcher,
    root_dir: Option<&Path>,
) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !dir.exists() {
        return files;
    }

    let root = root_dir.unwrap_or(dir);

    let Ok(entries) = std::fs::read_dir(dir) else {
        return files;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if skip_node_modules && name == "node_modules" {
            continue;
        }

        let full_path = entry.path();
        let metadata = std::fs::metadata(&full_path);
        let is_dir = metadata.as_ref().is_ok_and(std::fs::Metadata::is_dir);
        let is_file = metadata.as_ref().is_ok_and(std::fs::Metadata::is_file);
        if metadata.is_err() {
            continue;
        }

        let relative = to_posix_path(
            &full_path
                .strip_prefix(root)
                .unwrap_or(&full_path)
                .to_string_lossy(),
        );
        let ignore_path = if is_dir {
            format!("{relative}/")
        } else {
            relative.clone()
        };
        if matcher.ignores(&ignore_path, is_dir) {
            continue;
        }

        if is_dir {
            files.extend(collect_files(
                &full_path,
                resource_type,
                skip_node_modules,
                matcher,
                Some(root),
            ));
        } else if is_file && file_matches(resource_type, &full_path) {
            files.push(full_path);
        }
    }
    files
}

/// The skill discovery modes, upstream's `SkillDiscoveryMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillDiscoveryMode {
    /// Agent-dir skills: root markdown files count, upstream's `"pi"`.
    Pi,
    /// `.agents/skills` directories: only nested skills count, upstream's
    /// `"agents"`.
    Agents,
}

/// Walk a directory collecting skill entries, upstream's
/// `collectSkillEntries`: the first `SKILL.md` in the directory wins and
/// stops the walk under it; a nested directory recurses; root-level
/// markdown files count only in `pi` mode; dot entries and `node_modules`
/// skip.
fn collect_skill_entries(
    dir: &Path,
    mode: SkillDiscoveryMode,
    matcher: &IgnoreMatcher,
    root_dir: Option<&Path>,
) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    if !dir.exists() {
        return entries;
    }

    let root = root_dir.unwrap_or(dir);

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        return entries;
    };
    let collected: Vec<_> = dir_entries.flatten().collect();

    for entry in &collected {
        if entry.file_name() != "SKILL.md" {
            continue;
        }
        let full_path = entry.path();
        let metadata = std::fs::metadata(&full_path);
        let Some(metadata) = metadata.ok() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let relative = to_posix_path(
            &full_path
                .strip_prefix(root)
                .unwrap_or(&full_path)
                .to_string_lossy(),
        );
        if !matcher.ignores(&relative, false) {
            entries.push(full_path);
            return entries;
        }
    }

    for entry in &collected {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let full_path = entry.path();
        let metadata = std::fs::metadata(&full_path);
        let Some(metadata) = metadata.ok() else {
            continue;
        };
        let is_dir = metadata.is_dir();
        let is_file = metadata.is_file();

        let relative = to_posix_path(
            &full_path
                .strip_prefix(root)
                .unwrap_or(&full_path)
                .to_string_lossy(),
        );
        #[expect(
            clippy::case_sensitive_file_extension_comparisons,
            reason = "upstream's `.endsWith(\".md\")` is case-sensitive; matching `.MD` would load skill entries upstream skips"
        )]
        let should_include_markdown = is_file
            && name.ends_with(".md")
            && !matcher.ignores(&relative, false)
            && match mode {
                SkillDiscoveryMode::Pi => dir == root,
                SkillDiscoveryMode::Agents => dir != root,
            };
        if should_include_markdown {
            entries.push(full_path);
            continue;
        }

        if !is_dir {
            continue;
        }
        if matcher.ignores(&format!("{relative}/"), true) {
            continue;
        }

        entries.extend(collect_skill_entries(&full_path, mode, matcher, Some(root)));
    }
    entries
}

fn collect_auto_skill_entries(dir: &Path, mode: SkillDiscoveryMode) -> Vec<PathBuf> {
    collect_skill_entries(dir, mode, &build_ignore_matcher(dir, dir), None)
}

/// The git repository root walking up from a directory, upstream's
/// `findGitRepoRoot`: the nearest ancestor holding a `.git` entry.
pub(crate) fn find_git_repo_root(start_dir: &Path) -> Option<PathBuf> {
    let mut dir = start_dir.to_path_buf();
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        let parent = dir.parent()?;
        if parent == dir {
            return None;
        }
        dir = parent.to_path_buf();
    }
}

/// The `.agents/skills` directories from the cwd up to the git root,
/// upstream's `collectAncestorAgentsSkillDirs` — the walk stops at the repo
/// root (or the filesystem root when outside a repo).
pub(crate) fn collect_ancestor_agents_skill_dirs(start_dir: &Path) -> Vec<PathBuf> {
    let mut skill_dirs = Vec::new();
    let resolved_start_dir = start_dir.to_path_buf();
    let git_repo_root = find_git_repo_root(&resolved_start_dir);

    let mut dir = resolved_start_dir;
    loop {
        skill_dirs.push(dir.join(".agents").join("skills"));
        if git_repo_root.as_ref().is_some_and(|root| *root == dir) {
            break;
        }
        match dir.parent() {
            Some(parent) if parent != dir => dir = parent.to_path_buf(),
            _ => break,
        }
    }
    skill_dirs
}

/// Walk a directory collecting prompt markdown files, upstream's
/// `collectAutoPromptEntries`.
fn collect_auto_prompt_entries(dir: &Path) -> Vec<PathBuf> {
    collect_extension_leaf_files(dir, ResourceType::Prompts)
}

/// Walk a directory collecting theme JSON files, upstream's
/// `collectAutoThemeEntries`.
fn collect_auto_theme_entries(dir: &Path) -> Vec<PathBuf> {
    collect_extension_leaf_files(dir, ResourceType::Themes)
}

/// The leaf-file walk `collectAutoPromptEntries`/`collectAutoThemeEntries`
/// share: one level of files matching the filter, dot entries and
/// `node_modules` skipped, ignore files honored.
fn collect_extension_leaf_files(dir: &Path, resource_type: ResourceType) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    if !dir.exists() {
        return entries;
    }

    let matcher = build_ignore_matcher(dir, dir);

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        return entries;
    };
    for entry in dir_entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let full_path = entry.path();
        let metadata = std::fs::metadata(&full_path);
        let Some(metadata) = metadata.ok() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }

        let relative = to_posix_path(
            &full_path
                .strip_prefix(dir)
                .unwrap_or(&full_path)
                .to_string_lossy(),
        );
        if matcher.ignores(&relative, false) {
            continue;
        }
        if file_matches(resource_type, &full_path) {
            entries.push(full_path);
        }
    }
    entries
}

/// Resolve a manifest entry against its package root, upstream's
/// `resolve(dir, extPath)`: the `./` prefix folds away and `..` segments
/// consume lexically.
fn normalize_manifest_entry(dir: &Path, entry: &str) -> PathBuf {
    let normalized = entry.replace('\\', "/");
    let mut resolved = dir.to_path_buf();
    for segment in normalized.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                resolved.pop();
            }
            other => resolved.push(other),
        }
    }
    resolved
}

/// The explicit extension entries a directory declares, upstream's
/// `resolveExtensionEntries`: the manifest's `pi.extensions` that exist,
/// else the executable `index` convention (upstream's `index.ts`/
/// `index.js`), else the `bin/` convention directory's executables (the
/// `crate:` channel's cargo-installed layout), else `None`.
fn resolve_extension_entries(dir: &Path) -> Option<Vec<PathBuf>> {
    let package_json_path = dir.join("package.json");
    if package_json_path.exists()
        && let Some(manifest) = crate::pi_manifest::read_pi_manifest(&package_json_path)
        && let Some(entries) = manifest.extensions.filter(|entries| !entries.is_empty())
    {
        let resolved: Vec<PathBuf> = entries
            .iter()
            .map(|ext_path| normalize_manifest_entry(dir, ext_path))
            .filter(|resolved_ext_path| resolved_ext_path.exists())
            .collect();
        if !resolved.is_empty() {
            return Some(resolved);
        }
    }

    let index = dir.join("index");
    if index.exists() && is_executable(&index) {
        return Some(vec![index]);
    }

    let bin_dir = dir.join("bin");
    if bin_dir.is_dir() {
        let executables: Vec<PathBuf> = std::fs::read_dir(&bin_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && is_executable(path))
            .collect();
        if !executables.is_empty() {
            return Some(executables);
        }
    }

    None
}

/// Discover extension entries from a directory's contents, upstream's
/// `collectAutoExtensionEntries`: the directory's own explicit entries win,
/// else each entry contributes — an executable file is an extension (the
/// restated `.ts`/`.js` rule), a directory contributes its explicit entries.
fn collect_auto_extension_entries(dir: &Path) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    if !dir.exists() {
        return entries;
    }

    if let Some(root_entries) = resolve_extension_entries(dir) {
        return root_entries;
    }

    let matcher = build_ignore_matcher(dir, dir);

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        return entries;
    };
    for entry in dir_entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let full_path = entry.path();
        let metadata = std::fs::metadata(&full_path);
        let Some(metadata) = metadata.ok() else {
            continue;
        };
        let is_dir = metadata.is_dir();
        let is_file = metadata.is_file();

        let relative = to_posix_path(
            &full_path
                .strip_prefix(dir)
                .unwrap_or(&full_path)
                .to_string_lossy(),
        );
        if matcher.ignores(&relative, is_dir) {
            continue;
        }

        if is_file && is_executable(&full_path) {
            entries.push(full_path);
        } else if is_dir && let Some(resolved_entries) = resolve_extension_entries(&full_path) {
            entries.extend(resolved_entries);
        }
    }
    entries
}

/// Collect a directory's resource files per type, upstream's
/// `collectResourceFiles`: skills walk their `pi`-mode rules, extensions
/// use the smart discovery, the rest walk their file patterns.
fn collect_resource_files(dir: &Path, resource_type: ResourceType) -> Vec<PathBuf> {
    match resource_type {
        ResourceType::Skills => collect_skill_entries(
            dir,
            SkillDiscoveryMode::Pi,
            &build_ignore_matcher(dir, dir),
            None,
        ),
        ResourceType::Extensions => collect_auto_extension_entries(dir),
        ResourceType::Prompts | ResourceType::Themes => collect_files(
            dir,
            resource_type,
            true,
            &build_ignore_matcher(dir, dir),
            None,
        ),
    }
}

/// Whether a file matches any include pattern, upstream's
/// `matchesAnyPattern`: the relative path, the file name, and the absolute
/// posix path each get a minimatch chance; `SKILL.md` files also let their
/// parent directory match.
fn matches_any_pattern(file_path: &Path, patterns: &[String], base_dir: &Path) -> bool {
    let rel = to_posix_path(
        &file_path
            .strip_prefix(base_dir)
            .unwrap_or(file_path)
            .to_string_lossy(),
    );
    let name = file_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let file_path_posix = to_posix_path(&file_path.to_string_lossy());
    let is_skill_file = name == "SKILL.md";
    let parent_dir = is_skill_file.then(|| {
        file_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default()
    });
    let parent_rel = parent_dir.as_ref().map(|parent| {
        to_posix_path(
            &parent
                .strip_prefix(base_dir)
                .unwrap_or(parent)
                .to_string_lossy(),
        )
    });
    let parent_name = parent_dir.as_ref().map(|parent| {
        parent
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    let parent_dir_posix = parent_dir
        .as_ref()
        .map(|parent| to_posix_path(&parent.to_string_lossy()));

    patterns.iter().any(|pattern| {
        let normalized_pattern = to_posix_path(pattern);
        if crate::utils::minimatch::matches(&rel, &normalized_pattern, false)
            || crate::utils::minimatch::matches(&name, &normalized_pattern, false)
            || crate::utils::minimatch::matches(&file_path_posix, &normalized_pattern, false)
        {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        [
            parent_rel.as_deref(),
            parent_name.as_deref(),
            parent_dir_posix.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|candidate| crate::utils::minimatch::matches(candidate, &normalized_pattern, false))
    })
}

/// Normalize an exact-entry pattern, upstream's `normalizeExactPattern`:
/// a `./`/`.\` prefix strips and the separators go posix.
fn normalize_exact_pattern(pattern: &str) -> String {
    let normalized = pattern
        .strip_prefix("./")
        .or_else(|| pattern.strip_prefix(".\\"))
        .unwrap_or(pattern);
    to_posix_path(normalized)
}

/// Whether a file matches an exact entry, upstream's
/// `matchesAnyExactPattern`: normalized equality against the relative and
/// absolute posix forms; `SKILL.md` files also match their parent
/// directory's forms.
fn matches_any_exact_pattern(file_path: &Path, patterns: &[String], base_dir: &Path) -> bool {
    if patterns.is_empty() {
        return false;
    }
    let rel = to_posix_path(
        &file_path
            .strip_prefix(base_dir)
            .unwrap_or(file_path)
            .to_string_lossy(),
    );
    let name = file_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let file_path_posix = to_posix_path(&file_path.to_string_lossy());
    let is_skill_file = name == "SKILL.md";
    let parent_dir = is_skill_file.then(|| {
        file_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default()
    });
    let parent_rel = parent_dir.as_ref().map(|parent| {
        to_posix_path(
            &parent
                .strip_prefix(base_dir)
                .unwrap_or(parent)
                .to_string_lossy(),
        )
    });
    let parent_dir_posix = parent_dir
        .as_ref()
        .map(|parent| to_posix_path(&parent.to_string_lossy()));

    patterns.iter().any(|pattern| {
        let normalized = normalize_exact_pattern(pattern);
        if normalized == rel || normalized == file_path_posix {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        [parent_rel.as_deref(), parent_dir_posix.as_deref()]
            .into_iter()
            .flatten()
            .any(|candidate| normalized == candidate)
    })
}

fn get_override_patterns(entries: &[String]) -> Vec<String> {
    entries
        .iter()
        .filter(|pattern| {
            pattern.starts_with('!') || pattern.starts_with('+') || pattern.starts_with('-')
        })
        .cloned()
        .collect()
}

/// Whether the auto-discovered path stays enabled under the overrides,
/// upstream's `isEnabledByOverrides`: exclusions disable, force-includes
/// re-enable, force-excludes disable last.
fn is_enabled_by_overrides(file_path: &Path, patterns: &[String], base_dir: &Path) -> bool {
    let overrides = get_override_patterns(patterns);
    let excludes: Vec<String> = overrides
        .iter()
        .filter(|pattern| pattern.starts_with('!'))
        .map(|pattern| pattern[1..].to_string())
        .collect();
    let force_includes: Vec<String> = overrides
        .iter()
        .filter(|pattern| pattern.starts_with('+'))
        .map(|pattern| pattern[1..].to_string())
        .collect();
    let force_excludes: Vec<String> = overrides
        .iter()
        .filter(|pattern| pattern.starts_with('-'))
        .map(|pattern| pattern[1..].to_string())
        .collect();

    let excludes_hit = !excludes.is_empty() && matches_any_pattern(file_path, &excludes, base_dir);
    let force_include_hit = !force_includes.is_empty()
        && matches_any_exact_pattern(file_path, &force_includes, base_dir);
    let force_exclude_hit = !force_excludes.is_empty()
        && matches_any_exact_pattern(file_path, &force_excludes, base_dir);
    (!excludes_hit || force_include_hit) && !force_exclude_hit
}

/// Apply include/exclude/force patterns to paths, upstream's
/// `applyPatterns`: includes select, excludes remove, `+path` force-adds
/// from the full set, `-path` force-removes last.
fn apply_patterns(all_paths: &[PathBuf], patterns: &[String], base_dir: &Path) -> HashSet<PathBuf> {
    let mut includes = Vec::new();
    let mut excludes = Vec::new();
    let mut force_includes = Vec::new();
    let mut force_excludes = Vec::new();

    for pattern in patterns {
        if let Some(rest) = pattern.strip_prefix('+') {
            force_includes.push(rest.to_string());
        } else if let Some(rest) = pattern.strip_prefix('-') {
            force_excludes.push(rest.to_string());
        } else if let Some(rest) = pattern.strip_prefix('!') {
            excludes.push(rest.to_string());
        } else {
            includes.push(pattern.clone());
        }
    }

    let mut result: Vec<PathBuf> = if includes.is_empty() {
        all_paths.to_vec()
    } else {
        all_paths
            .iter()
            .filter(|file_path| matches_any_pattern(file_path, &includes, base_dir))
            .cloned()
            .collect()
    };

    if !excludes.is_empty() {
        result.retain(|file_path| !matches_any_pattern(file_path, &excludes, base_dir));
    }

    if !force_includes.is_empty() {
        for file_path in all_paths {
            if !result.contains(file_path)
                && matches_any_exact_pattern(file_path, &force_includes, base_dir)
            {
                result.push(file_path.clone());
            }
        }
    }

    if !force_excludes.is_empty() {
        result.retain(|file_path| !matches_any_exact_pattern(file_path, &force_excludes, base_dir));
    }

    result.into_iter().collect()
}

/// Apply autoload-disabled delta patterns, upstream's
/// `applyAutoloadDisabledPatterns`: each pattern flips the enabled state of
/// every matching path — `+`/`-` are exact, `!` disables by glob, a plain
/// pattern enables by glob.
fn apply_autoload_disabled_patterns(
    all_paths: &[PathBuf],
    patterns: &[String],
    base_dir: &Path,
) -> indexmap::IndexMap<PathBuf, bool> {
    let mut result = indexmap::IndexMap::new();
    for pattern in patterns {
        let strip = |prefix: char| pattern.strip_prefix(prefix).map(str::to_string);
        let target = strip('+')
            .or_else(|| strip('-'))
            .or_else(|| strip('!'))
            .unwrap_or_else(|| pattern.clone());
        let enabled = !pattern.starts_with('-') && !pattern.starts_with('!');
        let exact = pattern.starts_with('+') || pattern.starts_with('-');
        for file_path in all_paths {
            let matched = if exact {
                matches_any_exact_pattern(file_path, std::slice::from_ref(&target), base_dir)
            } else {
                matches_any_pattern(file_path, std::slice::from_ref(&target), base_dir)
            };
            if matched {
                result.insert(file_path.clone(), enabled);
            }
        }
    }
    result
}
// =============================================================================
// Command runner seam
// =============================================================================

/// The spawn options one runner call carries, upstream's per-call object
/// literals.
#[derive(Debug, Clone, Default)]
pub struct CommandRunOptions {
    /// The working directory, upstream's `cwd`.
    pub cwd: Option<String>,
    /// Extra environment merged over the inherited environment, upstream's
    /// `env` object spread.
    pub env: Vec<(String, String)>,
}

/// The child-process seam, upstream's private `runCommand`/`runCommandCapture`/
/// `runCommandSync` methods lifted so tests can inject a recording double
/// (upstream's `vi.spyOn`).
pub trait CommandRunner: Send + Sync {
    /// Run to completion with inherited stdio, upstream's `runCommand`.
    ///
    /// # Errors
    /// A spawn failure, or the child's non-zero exit with the upstream
    /// message shape.
    fn run(
        &self,
        command: &str,
        args: &[String],
        options: &CommandRunOptions,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<(), PackageManagerError>> + Send + '_>>;

    /// Run with both pipes captured, upstream's `runCommandCapture`.
    ///
    /// # Errors
    /// A spawn failure, the timeout's kill, or the child's failure with the
    /// upstream message shape; success resolves the trimmed stdout.
    fn run_capture(
        &self,
        command: &str,
        args: &[String],
        options: &CommandRunOptions,
        timeout_ms: Option<u64>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, PackageManagerError>> + Send + '_>>;

    /// Run synchronously with both pipes captured, upstream's
    /// `runCommandSync`.
    ///
    /// # Errors
    /// The upstream `Failed to run <cmd> <args>: <reason>` message.
    fn run_sync(&self, command: &str, args: &[String]) -> Result<String, PackageManagerError>;
}

/// The real runner, upstream's `spawnProcess`/`spawnProcessSync` wiring.
#[derive(Debug, Default)]
pub struct ProcessCommandRunner;

impl CommandRunner for ProcessCommandRunner {
    fn run(
        &self,
        command: &str,
        args: &[String],
        options: &CommandRunOptions,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<(), PackageManagerError>> + Send + '_>> {
        let command = command.to_string();
        let args = args.to_vec();
        let options = options.clone();
        Box::pin(async move {
            let mut cmd = tokio::process::Command::new(&command);
            cmd.args(&args)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit());
            if let Some(cwd) = options.cwd.as_deref() {
                cmd.current_dir(cwd);
            }
            for (name, value) in &options.env {
                cmd.env(name, value);
            }
            let mut child = cmd
                .spawn()
                .map_err(|error| PackageManagerError(error.to_string()))?;
            let status = child
                .wait()
                .await
                .map_err(|error| PackageManagerError(error.to_string()))?;
            match status.code() {
                Some(0) => Ok(()),
                Some(code) => Err(PackageManagerError(format!(
                    "{} {} failed with code {code}",
                    command,
                    args.join(" ")
                ))),
                None => Err(PackageManagerError(format!(
                    "{} {} failed with an unknown exit status",
                    command,
                    args.join(" ")
                ))),
            }
        })
    }

    fn run_capture(
        &self,
        command: &str,
        args: &[String],
        options: &CommandRunOptions,
        timeout_ms: Option<u64>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, PackageManagerError>> + Send + '_>>
    {
        let command = command.to_string();
        let args = args.to_vec();
        let options = options.clone();
        Box::pin(async move {
            let mut cmd = tokio::process::Command::new(&command);
            cmd.args(&args)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            if let Some(cwd) = options.cwd.as_deref() {
                cmd.current_dir(cwd);
            }
            for (name, value) in &options.env {
                cmd.env(name, value);
            }
            let mut child = cmd
                .spawn()
                .map_err(|error| PackageManagerError(error.to_string()))?;
            let mut stdout_pipe = child.stdout.take();
            let mut stderr_pipe = child.stderr.take();
            let stdout_task = tokio::spawn(async move {
                let mut buffer = Vec::new();
                if let Some(pipe) = stdout_pipe.as_mut() {
                    use tokio::io::AsyncReadExt;
                    let _ = pipe.read_to_end(&mut buffer).await;
                }
                buffer
            });
            let stderr_task = tokio::spawn(async move {
                let mut buffer = Vec::new();
                if let Some(pipe) = stderr_pipe.as_mut() {
                    use tokio::io::AsyncReadExt;
                    let _ = pipe.read_to_end(&mut buffer).await;
                }
                buffer
            });

            let status = if let Some(timeout_ms) = timeout_ms {
                tokio::time::timeout(Duration::from_millis(timeout_ms), child.wait())
                    .await
                    .map_err(|_| {
                        PackageManagerError(format!(
                            "{} {} timed out after {timeout_ms}ms",
                            command,
                            args.join(" ")
                        ))
                    })?
                    .map_err(|error| PackageManagerError(error.to_string()))?
            } else {
                child
                    .wait()
                    .await
                    .map_err(|error| PackageManagerError(error.to_string()))?
            };
            let stdout = stdout_task
                .await
                .map(|buffer| String::from_utf8_lossy(&buffer).into_owned())
                .unwrap_or_default();
            let stderr = stderr_task
                .await
                .map(|buffer| String::from_utf8_lossy(&buffer).into_owned())
                .unwrap_or_default();

            if status.code() == Some(0) {
                return Ok(stdout.trim().to_string());
            }
            let exit_status = status.code().map_or_else(
                || format!("signal {}", signal_name(status)),
                |code| format!("code {code}"),
            );
            let detail = if stderr.trim().is_empty() {
                stdout
            } else {
                stderr
            };
            Err(PackageManagerError(format!(
                "{} {} failed with {exit_status}: {}",
                command,
                args.join(" "),
                detail
            )))
        })
    }

    fn run_sync(&self, command: &str, args: &[String]) -> Result<String, PackageManagerError> {
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let result = crate::utils::child_process::spawn_process_sync(
            command,
            &arg_refs,
            &crate::utils::child_process::SpawnSyncOptions {
                capture_output: true,
                capture_stderr: true,
                timeout_ms: None,
            },
        );
        if result.status != Some(0) {
            let reason = if result.stderr.trim().is_empty() {
                &result.stdout
            } else {
                &result.stderr
            };
            return Err(PackageManagerError(format!(
                "Failed to run {} {}: {}",
                command,
                args.join(" "),
                reason.trim()
            )));
        }
        let output = if result.stdout.is_empty() {
            &result.stderr
        } else {
            &result.stdout
        };
        Ok(output.trim().to_string())
    }
}

use std::future::Future;
use std::time::Duration;

/// The signal name a terminated status carries, upstream's `signal` event
/// argument restated to the signal number (Rust's exit status carries no
/// name table).
#[cfg(unix)]
pub(crate) fn signal_name(status: std::process::ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    status
        .signal()
        .map_or_else(|| "unknown".to_string(), |signal| signal.to_string())
}

#[cfg(not(unix))]
fn signal_name(_status: std::process::ExitStatus) -> String {
    "unknown".to_string()
}

// =============================================================================
// Manager
// =============================================================================

/// The manager's construction options, upstream's `PackageManagerOptions`
/// plus the seams the tests inject.
pub struct PackageManagerOptions<S: SettingsStorage> {
    /// The working directory, upstream's `cwd`.
    pub cwd: String,
    /// The agent directory, upstream's `agentDir`.
    pub agent_dir: String,
    /// The settings manager, upstream's `settingsManager` handle.
    pub settings: Arc<Mutex<SettingsManager<S>>>,
    /// The child-process runner; the default spawns real processes.
    pub command_runner: Option<Arc<dyn CommandRunner>>,
    /// The environment lookup; the default reads the real environment.
    pub env: Option<EnvLookup>,
    /// The HTTP client the registry checks and tarball downloads ride.
    pub http_client: Option<Arc<dyn HttpClient>>,
}

/// The package manager, upstream's `DefaultPackageManager`.
pub struct DefaultPackageManager<S: SettingsStorage> {
    cwd: PathBuf,
    agent_dir: PathBuf,
    settings: Arc<Mutex<SettingsManager<S>>>,
    command_runner: Arc<dyn CommandRunner>,
    env: EnvLookup,
    http_client: Arc<dyn HttpClient>,
    progress_callback: Mutex<Option<ProgressCallback>>,
}

/// The missing-source callback, upstream's
/// `(source: string) => Promise<MissingSourceAction>` restated as a sync
/// closure — the interactive prompt riding it is the caller's concern.
pub type ResolveMissingCallback =
    Box<dyn Fn(&str) -> Result<MissingSourceAction, PackageManagerError> + Send + Sync>;

/// A settings entry, upstream's `PackageSource` union restated as the view
/// the raw settings values parse to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageSourceView {
    /// A plain source string, upstream's string form.
    Source(String),
    /// An object entry: the source plus its filters, upstream's object form.
    Filtered(PackageFilter),
}

/// The object entry's filters, upstream's
/// `{ source, autoload?, extensions?, skills?, prompts?, themes? }`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PackageFilter {
    /// The source string, upstream's `source`.
    pub source: String,
    /// Whether the package auto-loads all resources, upstream's
    /// `autoload?`; `false` turns the entry into a delta over the
    /// same-source user entry.
    pub autoload: Option<bool>,
    /// The extension filters, upstream's `extensions?`.
    pub extensions: Option<Vec<String>>,
    /// The skill filters, upstream's `skills?`.
    pub skills: Option<Vec<String>>,
    /// The prompt filters, upstream's `prompts?`.
    pub prompts: Option<Vec<String>>,
    /// The theme filters, upstream's `themes?`.
    pub themes: Option<Vec<String>>,
}

impl PackageSourceView {
    /// Parse a raw settings value, upstream's `typeof pkg === "string"` /
    /// `typeof pkg === "object"` narrowing: strings pass through, objects
    /// read their `source` string and string-array fields, anything else
    /// reads as a plain string entry (upstream trusts the type and would
    /// throw on `undefined` — the raw entry rides its `String` coercion).
    #[must_use]
    pub fn parse(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(source) => Self::Source(source.clone()),
            serde_json::Value::Object(object) => {
                let source = object
                    .get("source")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let read_strings = |key: &str| -> Option<Vec<String>> {
                    object
                        .get(key)
                        .and_then(serde_json::Value::as_array)
                        .map(|entries| {
                            entries
                                .iter()
                                .filter_map(serde_json::Value::as_str)
                                .map(str::to_string)
                                .collect()
                        })
                };
                Self::Filtered(PackageFilter {
                    source,
                    autoload: object.get("autoload").and_then(serde_json::Value::as_bool),
                    extensions: read_strings("extensions"),
                    skills: read_strings("skills"),
                    prompts: read_strings("prompts"),
                    themes: read_strings("themes"),
                })
            }
            other => Self::Source(other.to_string()),
        }
    }

    /// The source string, upstream's `getPackageSourceString`.
    #[must_use]
    pub fn source(&self) -> &str {
        match self {
            Self::Source(source) => source,
            Self::Filtered(filter) => &filter.source,
        }
    }

    /// The filters when the entry carries them, upstream's
    /// `typeof pkg === "object" ? pkg : undefined`.
    #[must_use]
    pub const fn filter(&self) -> Option<&PackageFilter> {
        match self {
            Self::Source(_) => None,
            Self::Filtered(filter) => Some(filter),
        }
    }

    /// The raw value to write back, upstream's entry identity in the
    /// settings arrays.
    #[must_use]
    pub fn to_value(&self) -> serde_json::Value {
        match self {
            Self::Source(source) => serde_json::Value::String(source.clone()),
            Self::Filtered(filter) => {
                let mut object = serde_json::Map::new();
                object.insert(
                    "source".to_string(),
                    serde_json::Value::String(filter.source.clone()),
                );
                if let Some(autoload) = filter.autoload {
                    object.insert("autoload".to_string(), serde_json::Value::Bool(autoload));
                }
                for (key, value) in [
                    ("extensions", &filter.extensions),
                    ("skills", &filter.skills),
                    ("prompts", &filter.prompts),
                    ("themes", &filter.themes),
                ] {
                    if let Some(entries) = value {
                        object.insert(
                            key.to_string(),
                            serde_json::Value::Array(
                                entries
                                    .iter()
                                    .map(|entry| serde_json::Value::String(entry.clone()))
                                    .collect(),
                            ),
                        );
                    }
                }
                serde_json::Value::Object(object)
            }
        }
    }
}

/// A settings entry with its scope, upstream's
/// `Array<{ pkg: PackageSource; scope: SourceScope }>` entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopedSource {
    /// The parsed entry, upstream's `pkg`.
    pub pkg: PackageSourceView,
    /// The scope it is configured at, upstream's `scope`.
    pub scope: SourceScope,
}

/// Resolve a settings-relative input path through the path belt, upstream's
/// private `resolvePath`/`resolvePathFromBase` pair: tilde expansion over
/// the given home, the base as the relative root, `trim` when set. The
/// `file://` URL failure node throws degrades to the raw input — the
/// manager's inputs are process paths, never URLs.
fn resolve_input_path(input: &str, base_dir: &str, home: &str, trim: bool) -> PathBuf {
    let options = crate::utils::paths::PathInputOptions {
        trim,
        home_dir: Some(home.to_string()),
        ..crate::utils::paths::PathInputOptions::default()
    };
    crate::utils::paths::resolve_path_with(input, base_dir, &options)
        .map_or_else(|_| PathBuf::from(input), PathBuf::from)
}

impl<S: SettingsStorage + 'static> DefaultPackageManager<S> {
    /// Build the manager, upstream's constructor.
    pub fn new(options: PackageManagerOptions<S>) -> Self {
        let home = options
            .env
            .as_ref()
            .map_or_else(crate::config::home_dir, get_home_dir);
        Self {
            // The constructor's resolves ride the default options: no trim,
            // the process home for tilde expansion (upstream's
            // `resolvePath(options.cwd)` with `process.cwd()` as base).
            cwd: resolve_input_path(&options.cwd, &crate::config::process_cwd(), &home, false),
            agent_dir: resolve_input_path(
                &options.agent_dir,
                &crate::config::process_cwd(),
                &home,
                false,
            ),
            settings: options.settings,
            command_runner: options
                .command_runner
                .unwrap_or_else(|| Arc::new(ProcessCommandRunner)),
            env: options
                .env
                .unwrap_or_else(crate::config::default_env_lookup),
            http_client: options
                .http_client
                .unwrap_or_else(pi_ai::http::default_http_client),
            progress_callback: Mutex::new(None),
        }
    }

    /// Set the progress callback, upstream's `setProgressCallback`.
    pub fn set_progress_callback(&mut self, callback: Option<ProgressCallback>) {
        *self
            .progress_callback
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = callback;
    }

    fn emit_progress(&self, event: &ProgressEvent) {
        if let Some(callback) = self
            .progress_callback
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            callback(event);
        }
    }

    async fn with_progress(
        &self,
        action: ProgressAction,
        source: &str,
        message: &str,
        operation: impl Future<Output = Result<(), PackageManagerError>>,
    ) -> Result<(), PackageManagerError> {
        self.emit_progress(&ProgressEvent {
            event_type: ProgressEventType::Start,
            action,
            source: source.to_string(),
            message: Some(message.to_string()),
        });
        match operation.await {
            Ok(()) => {
                self.emit_progress(&ProgressEvent {
                    event_type: ProgressEventType::Complete,
                    action,
                    source: source.to_string(),
                    message: None,
                });
                Ok(())
            }
            Err(error) => {
                self.emit_progress(&ProgressEvent {
                    event_type: ProgressEventType::Error,
                    action,
                    source: source.to_string(),
                    message: Some(error.0.clone()),
                });
                Err(error)
            }
        }
    }

    fn settings_lock(&self) -> std::sync::MutexGuard<'_, SettingsManager<S>> {
        self.settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn offline(&self) -> bool {
        is_offline_mode_enabled_with(&self.env)
    }

    fn home(&self) -> String {
        get_home_dir(&self.env)
    }

    /// Resolve a settings-relative input path, upstream's `resolvePath`
    /// private helper: the cwd as the base, `trim` on, the injected home
    /// for tilde expansion.
    fn resolve_path(&self, input: &str) -> String {
        resolve_input_path(input, &self.cwd.to_string_lossy(), &self.home(), true)
            .to_string_lossy()
            .into_owned()
    }

    /// Resolve against an explicit base, upstream's `resolvePathFromBase`.
    fn resolve_path_from_base(&self, input: &str, base_dir: &str) -> String {
        resolve_input_path(input, base_dir, &self.home(), true)
            .to_string_lossy()
            .into_owned()
    }

    /// The scope's settings-relative base, upstream's
    /// `getBaseDirForScope`: project → `<cwd>/<CONFIG_DIR_NAME>` (gated on
    /// trust), user → the agent dir, temporary → the cwd.
    fn get_base_dir_for_scope(&self, scope: SourceScope) -> Result<String, PackageManagerError> {
        match scope {
            SourceScope::Project => {
                self.assert_project_trusted_for_scope(scope)?;
                Ok(self
                    .cwd
                    .join(CONFIG_DIR_NAME)
                    .to_string_lossy()
                    .into_owned())
            }
            SourceScope::User => Ok(self.agent_dir.to_string_lossy().into_owned()),
            SourceScope::Temporary => Ok(self.cwd.to_string_lossy().into_owned()),
        }
    }

    /// The project-trust gate, upstream's `assertProjectTrustedForScope`.
    fn assert_project_trusted_for_scope(
        &self,
        scope: SourceScope,
    ) -> Result<(), PackageManagerError> {
        if scope == SourceScope::Project && !self.settings_lock().is_project_trusted() {
            return Err(PackageManagerError(
                "Project is not trusted; refusing to access project package storage".to_string(),
            ));
        }
        Ok(())
    }

    /// Add a source to settings, upstream's `addSourceToSettings`.
    ///
    /// # Errors
    /// A settings write failure, or a parse failure of the source.
    pub fn add_source_to_settings(
        &self,
        source: &str,
        local: bool,
    ) -> Result<bool, PackageManagerError> {
        let scope = if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        let current_packages = self.configured_packages(scope);
        let normalized_source = self.normalize_package_source_for_settings(source, scope)?;
        let match_index = current_packages
            .iter()
            .position(|existing| self.package_sources_match(existing, source, scope));
        let next_packages: Vec<serde_json::Value> = if let Some(index) = match_index {
            let existing = &current_packages[index];
            if PackageSourceView::parse(existing).source() == normalized_source {
                return Ok(false);
            }
            let mut next: Vec<serde_json::Value> = current_packages.clone();
            next[index] = match existing {
                serde_json::Value::String(_) => serde_json::Value::String(normalized_source),
                serde_json::Value::Object(object) => {
                    let mut replaced = object.clone();
                    replaced.insert(
                        "source".to_string(),
                        serde_json::Value::String(normalized_source),
                    );
                    serde_json::Value::Object(replaced)
                }
                other => other.clone(),
            };
            next
        } else {
            let mut next: Vec<serde_json::Value> = current_packages;
            next.push(serde_json::Value::String(normalized_source));
            next
        };
        self.write_packages(scope, &next_packages)?;
        Ok(true)
    }

    /// Remove a source from settings, upstream's `removeSourceFromSettings`.
    ///
    /// # Errors
    /// A settings write failure.
    pub fn remove_source_from_settings(
        &self,
        source: &str,
        local: bool,
    ) -> Result<bool, PackageManagerError> {
        let scope = if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        let current_packages = self.configured_packages(scope);
        let next_packages: Vec<serde_json::Value> = current_packages
            .iter()
            .filter(|existing| !self.package_sources_match(existing, source, scope))
            .cloned()
            .collect();
        let changed = next_packages.len() != current_packages.len();
        if !changed {
            return Ok(false);
        }
        self.write_packages(scope, &next_packages)?;
        Ok(true)
    }

    /// The raw settings packages for a scope, upstream's
    /// `getGlobalSettings().packages` reads.
    fn configured_packages(&self, scope: SourceScope) -> Vec<serde_json::Value> {
        let manager = self.settings_lock();
        let settings = match scope {
            SourceScope::Project => manager.get_project_settings(),
            SourceScope::User | SourceScope::Temporary => manager.get_global_settings(),
        };
        let packages = settings
            .get("packages")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        drop(manager);
        packages
    }

    fn write_packages(
        &self,
        scope: SourceScope,
        packages: &[serde_json::Value],
    ) -> Result<(), PackageManagerError> {
        let mut manager = self.settings_lock();
        let result = match scope {
            SourceScope::Project => manager
                .set_project_packages(packages)
                .map_err(PackageManagerError),
            SourceScope::User | SourceScope::Temporary => {
                manager.set_packages(packages);
                Ok(())
            }
        };
        drop(manager);
        result
    }

    /// Whether an existing settings entry matches an input source, upstream's
    /// `packageSourcesMatch`: the settings-side key resolves against the
    /// entry's own scope base, the input-side key against the input scope.
    fn package_sources_match(
        &self,
        existing: &serde_json::Value,
        input_source: &str,
        scope: SourceScope,
    ) -> bool {
        let left = self
            .get_source_match_key_for_settings(PackageSourceView::parse(existing).source(), scope);
        let right = self.get_source_match_key_for_input(input_source);
        left == right
    }

    /// The identity key an input source matches on, upstream's
    /// `getSourceMatchKeyForInput`.
    fn get_source_match_key_for_input(&self, source: &str) -> String {
        match &self.parse_source(source) {
            Ok(ParsedSource::Crate(parsed)) => format!("crate:{}", parsed.name),
            Ok(ParsedSource::Git(parsed)) => format!("git:{}/{}", parsed.host, parsed.path),
            Ok(ParsedSource::Tarball(parsed)) => format!("tarball:{}", parsed.url),
            Ok(ParsedSource::Local(parsed)) => format!("local:{}", self.resolve_path(&parsed.path)),
            Err(_) => format!("local:{}", self.resolve_path(source)),
        }
    }

    /// The identity key a settings entry matches on, upstream's
    /// `getSourceMatchKeyForSettings`.
    fn get_source_match_key_for_settings(&self, source: &str, scope: SourceScope) -> String {
        match &self.parse_source(source) {
            Ok(ParsedSource::Crate(parsed)) => format!("crate:{}", parsed.name),
            Ok(ParsedSource::Git(parsed)) => format!("git:{}/{}", parsed.host, parsed.path),
            Ok(ParsedSource::Tarball(parsed)) => format!("tarball:{}", parsed.url),
            Ok(ParsedSource::Local(parsed)) => {
                let base_dir = self.get_base_dir_for_scope(scope).unwrap_or_default();
                format!(
                    "local:{}",
                    self.resolve_path_from_base(&parsed.path, &base_dir)
                )
            }
            Err(_) => {
                let base_dir = self.get_base_dir_for_scope(scope).unwrap_or_default();
                format!("local:{}", self.resolve_path_from_base(source, &base_dir))
            }
        }
    }

    /// Store a local source relative to its scope's settings base, upstream's
    /// `normalizePackageSourceForSettings`: the relative path, or `.` when
    /// the package sits at the base.
    fn normalize_package_source_for_settings(
        &self,
        source: &str,
        scope: SourceScope,
    ) -> Result<String, PackageManagerError> {
        if !matches!(self.parse_source(source)?, ParsedSource::Local(_)) {
            return Ok(source.to_string());
        }
        let base_dir = self.get_base_dir_for_scope(scope)?;
        let resolved = self.resolve_path(source);
        let relative = crate::utils::paths::get_cwd_relative_path(&resolved, &base_dir);
        Ok(relative.unwrap_or_else(|| {
            let stripped = resolved
                .strip_prefix(&base_dir)
                .unwrap_or(resolved.as_str());
            if stripped.is_empty() {
                ".".to_string()
            } else {
                stripped.to_string()
            }
        }))
    }

    /// The unique package identity ignoring version and ref, upstream's
    /// `getPackageIdentity` — host/path for git so SSH and HTTPS spellings
    /// collapse.
    fn get_package_identity(&self, source: &str, scope: Option<SourceScope>) -> String {
        match &self.parse_source(source) {
            Ok(ParsedSource::Crate(parsed)) => format!("crate:{}", parsed.name),
            Ok(ParsedSource::Git(parsed)) => format!("git:{}/{}", parsed.host, parsed.path),
            Ok(ParsedSource::Tarball(parsed)) => format!("tarball:{}", parsed.url),
            Ok(ParsedSource::Local(parsed)) => scope.map_or_else(
                || format!("local:{}", self.resolve_path(&parsed.path)),
                |scope| {
                    let base_dir = self.get_base_dir_for_scope(scope).unwrap_or_default();
                    format!(
                        "local:{}",
                        self.resolve_path_from_base(&parsed.path, &base_dir)
                    )
                },
            ),
            Err(_) => format!("local:{}", self.resolve_path(source)),
        }
    }

    /// Dedupe configured packages, upstream's `dedupePackages`: project
    /// scope wins; an `autoload: false` project entry is a delta over the
    /// user entry, so both stay (delta first). Project entries arrive
    /// before user entries in the resolve path.
    fn dedupe_packages(&self, packages: &[ScopedSource]) -> Vec<ScopedSource> {
        let mut result: Vec<ScopedSource> = Vec::new();
        let mut seen: HashMap<String, usize> = HashMap::new();
        for entry in packages {
            let identity = self.get_package_identity(entry.pkg.source(), Some(entry.scope));
            match seen.get(&identity) {
                None => {
                    seen.insert(identity, result.len());
                    result.push(entry.clone());
                }
                Some(&index) => {
                    let existing = &result[index];
                    if existing.scope == SourceScope::Project && entry.scope == SourceScope::User {
                        if existing
                            .pkg
                            .filter()
                            .is_some_and(|filter| filter.autoload == Some(false))
                        {
                            result.push(entry.clone());
                        }
                    } else if entry.scope == SourceScope::Project {
                        result[index] = entry.clone();
                    }
                }
            }
        }
        result
    }
}

// =============================================================================
// Source parsing
// =============================================================================

impl<S: SettingsStorage + 'static> DefaultPackageManager<S> {
    /// Parse a source string, upstream's `parseSource` over the dual-channel
    /// grammar: `crate:` compiles, git spellings (including `github:`)
    /// check out, tarball URLs download, everything local resolves in
    /// place.
    ///
    /// # Errors
    /// An `npm:` source — the npm channel drops per ADR 0007; settings
    /// carrying one must migrate to the new grammar (the import tool
    /// reports TS npm entries as skipped).
    pub fn parse_source(&self, source: &str) -> Result<ParsedSource, PackageManagerError> {
        if let Some(spec) = source.strip_prefix("npm:") {
            let _ = spec;
            return Err(PackageManagerError(
                "npm package sources are not supported; use crate:, github:, a tarball URL, or a local path"
                    .to_string(),
            ));
        }

        if let Some(spec) = source.strip_prefix("crate:") {
            return Ok(ParsedSource::Crate(parse_crate_spec(spec.trim())));
        }

        // The `github:` spelling is the git channel's canonical form; the
        // shorthand parses through the landed git-URL machinery so
        // identity, ref splitting, and validation match the other spellings.
        if let Some(shorthand) = source.strip_prefix("github:")
            && let Some(parsed) = parse_git_url(&format!("https://github.com/{shorthand}"))
        {
            return Ok(ParsedSource::Git(parsed));
        }

        if let Some(url) = tarball_url(source) {
            return Ok(ParsedSource::Tarball(TarballSource {
                url: url.to_string(),
            }));
        }

        if crate::utils::paths::is_local_path(source) {
            return Ok(ParsedSource::Local(LocalSource {
                path: source.to_string(),
            }));
        }

        if let Some(parsed) = parse_git_url(source) {
            return Ok(ParsedSource::Git(parsed));
        }

        Ok(ParsedSource::Local(LocalSource {
            path: source.to_string(),
        }))
    }
}

/// Parse a `crate:` spec into its name and optional version.
fn parse_crate_spec(spec: &str) -> CrateSource {
    match spec.split_once('@') {
        Some((name, version)) => CrateSource {
            name: name.to_string(),
            version: Some(version.to_string()),
            pinned: semver::Version::parse(version).is_ok(),
        },
        None => CrateSource {
            name: spec.to_string(),
            version: None,
            pinned: false,
        },
    }
}

/// Whether the string is a tarball URL: an `http(s)` URL whose path ends in
/// `.tgz` or `.tar.gz`. The npm channel's tarball spelling carried the
/// tarball inside the package spec; the Rust grammar names the URL.
///
/// The suffix comparisons ride the lowercased copy, so the case-sensitive
/// `ends_with` calls are already case-insensitive in effect.
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "the compared string is the lowercased copy above; the comparison is case-insensitive in effect and `Path::extension` would mishandle the `.tar.gz` double suffix"
)]
fn tarball_url(source: &str) -> Option<&str> {
    let lowered = source.to_ascii_lowercase();
    if !(lowered.starts_with("http://") || lowered.starts_with("https://")) {
        return None;
    }
    let path = lowered.split("://").nth(1)?;
    let path = path.split(['/', '?', '#']).next_back()?;
    if path.ends_with(".tgz") || path.ends_with(".tar.gz") {
        Some(source)
    } else {
        None
    }
}

// =============================================================================
// Resolve
// =============================================================================

/// Compute the precedence rank for a resource, upstream's
/// `resourcePrecedenceRank`: lower wins; project settings entries over
/// auto-discovered, user over packages.
fn resource_precedence_rank(metadata: &PathMetadata) -> i32 {
    if metadata.origin == ResourceOrigin::Package {
        return 4;
    }
    let scope_base = match metadata.scope {
        SourceScope::Project => 0,
        SourceScope::User | SourceScope::Temporary => 2,
    };
    scope_base + i32::from(metadata.source != "local")
}

impl<S: SettingsStorage + 'static> DefaultPackageManager<S> {
    /// Resolve every configured resource, upstream's `resolve`: packages
    /// first (project scope wins collisions), then the settings arrays,
    /// then auto-discovery.
    ///
    /// # Errors
    /// A package-source parse failure, the project-trust gate, or the
    /// missing-source callback's error.
    pub async fn resolve(
        &self,
        on_missing: Option<ResolveMissingCallback>,
    ) -> Result<ResolvedPaths, PackageManagerError> {
        let mut accumulator = create_accumulator();
        let global_settings = self.settings_lock().get_global_settings();
        let project_settings = self.settings_lock().get_project_settings();

        // Collect all packages with scope (project first so cwd resources
        // win collisions).
        let mut all_packages: Vec<ScopedSource> = Vec::new();
        for pkg in project_packages(&project_settings) {
            all_packages.push(ScopedSource {
                pkg,
                scope: SourceScope::Project,
            });
        }
        for pkg in project_packages(&global_settings) {
            all_packages.push(ScopedSource {
                pkg,
                scope: SourceScope::User,
            });
        }

        let package_sources = self.dedupe_packages(&all_packages);
        self.resolve_package_sources(package_sources, &mut accumulator, on_missing.as_ref())
            .await?;

        let global_base_dir = self.agent_dir.to_string_lossy().into_owned();
        let project_base_dir = self
            .cwd
            .join(CONFIG_DIR_NAME)
            .to_string_lossy()
            .into_owned();

        for resource_type in ResourceType::all() {
            let target = get_target_map(&mut accumulator, resource_type);
            let project_entries = settings_strings(&project_settings, resource_type.as_str());
            let global_entries = settings_strings(&global_settings, resource_type.as_str());
            self.resolve_local_entries(
                &project_entries,
                resource_type,
                target,
                &PathMetadata {
                    source: "local".to_string(),
                    scope: SourceScope::Project,
                    origin: ResourceOrigin::TopLevel,
                    base_dir: None,
                },
                &project_base_dir,
            );
            self.resolve_local_entries(
                &global_entries,
                resource_type,
                target,
                &PathMetadata {
                    source: "local".to_string(),
                    scope: SourceScope::User,
                    origin: ResourceOrigin::TopLevel,
                    base_dir: None,
                },
                &global_base_dir,
            );
        }

        self.add_auto_discovered_resources(
            &mut accumulator,
            &global_settings,
            &project_settings,
            &global_base_dir,
            &project_base_dir,
        );

        Ok(to_resolved_paths(accumulator))
    }

    /// Resolve one-off extension sources, upstream's
    /// `resolveExtensionSources`.
    ///
    /// # Errors
    /// A package-source parse failure, or the resolved install's
    /// path-computation failure.
    pub async fn resolve_extension_sources(
        &self,
        sources: &[String],
        local: bool,
        temporary: bool,
    ) -> Result<ResolvedPaths, PackageManagerError> {
        let mut accumulator = create_accumulator();
        let scope = if temporary {
            SourceScope::Temporary
        } else if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        let package_sources: Vec<ScopedSource> = sources
            .iter()
            .map(|source| ScopedSource {
                pkg: PackageSourceView::Source(source.clone()),
                scope,
            })
            .collect();
        self.resolve_package_sources(package_sources, &mut accumulator, None)
            .await?;
        Ok(to_resolved_paths(accumulator))
    }

    /// Walk the configured package sources, upstream's
    /// `resolvePackageSources`.
    #[expect(
        clippy::too_many_lines,
        reason = "the per-source-kind dispatch restates upstream's resolvePackageSources one-to-one; splitting it would scatter the channel branches"
    )]
    async fn resolve_package_sources(
        &self,
        sources: Vec<ScopedSource>,
        accumulator: &mut ResourceAccumulator,
        on_missing: Option<&ResolveMissingCallback>,
    ) -> Result<(), PackageManagerError> {
        for scoped in &sources {
            let source_str = scoped.pkg.source().to_string();
            let filter = scoped.pkg.filter().cloned();
            let delta_base = self.find_autoload_delta_base(&scoped.pkg, scoped.scope, &sources);
            let resolved_source = delta_base
                .as_ref()
                .map_or_else(|| source_str.clone(), |(source, _)| source.clone());
            let resolved_scope = delta_base.map_or(scoped.scope, |(_, scope)| scope);
            let parsed = self.parse_source(&resolved_source)?;
            let cloned_parsed = parsed.clone();
            let mut metadata = PathMetadata {
                source: source_str.clone(),
                scope: scoped.scope,
                origin: ResourceOrigin::Package,
                base_dir: None,
            };

            match parsed {
                ParsedSource::Local(local) => {
                    let base_dir = self.get_base_dir_for_scope(resolved_scope)?;
                    self.resolve_local_extension_source(
                        &local,
                        accumulator,
                        filter.as_ref(),
                        &metadata,
                        &base_dir,
                    );
                }
                ParsedSource::Crate(crate_source) => {
                    let mut installed_path =
                        self.get_crate_install_path(&crate_source, resolved_scope)?;
                    let needs_install = !Path::new(&installed_path).exists()
                        || !Self::installed_crate_matches_configured_version(
                            &crate_source,
                            &installed_path,
                        );
                    if needs_install {
                        if !self
                            .install_missing(
                                &resolved_source,
                                on_missing,
                                self.install_parsed_source(&cloned_parsed, resolved_scope),
                            )
                            .await?
                        {
                            continue;
                        }
                        installed_path =
                            self.get_crate_install_path(&crate_source, resolved_scope)?;
                    }
                    metadata.base_dir = Some(installed_path.clone());
                    self.collect_package_resources(
                        Path::new(&installed_path),
                        accumulator,
                        filter.as_ref(),
                        &metadata,
                    );
                }
                ParsedSource::Tarball(tarball) => {
                    let mut installed_path =
                        self.get_tarball_install_path(&tarball, resolved_scope)?;
                    if !Path::new(&installed_path).exists() {
                        if !self
                            .install_missing(
                                &resolved_source,
                                on_missing,
                                self.install_parsed_source(&cloned_parsed, resolved_scope),
                            )
                            .await?
                        {
                            continue;
                        }
                        installed_path = self.get_tarball_install_path(&tarball, resolved_scope)?;
                    }
                    metadata.base_dir = Some(installed_path.clone());
                    self.collect_package_resources(
                        Path::new(&installed_path),
                        accumulator,
                        filter.as_ref(),
                        &metadata,
                    );
                }
                ParsedSource::Git(git_source) => {
                    let installed_path = self.get_git_install_path(&git_source, resolved_scope)?;
                    if !Path::new(&installed_path).exists() {
                        if !self
                            .install_missing(
                                &resolved_source,
                                on_missing,
                                self.install_parsed_source(&cloned_parsed, resolved_scope),
                            )
                            .await?
                        {
                            continue;
                        }
                    } else if resolved_scope == SourceScope::Temporary
                        && !git_source.pinned
                        && !self.offline()
                    {
                        self.refresh_temporary_git_source(&git_source, &resolved_source)
                            .await;
                    }
                    metadata.base_dir = Some(installed_path.clone());
                    self.collect_package_resources(
                        Path::new(&installed_path),
                        accumulator,
                        filter.as_ref(),
                        &metadata,
                    );
                }
            }
        }
        Ok(())
    }

    /// The install-missing ladder, upstream's `installMissing`: offline
    /// skips, the callback decides, the default installs.
    async fn install_missing(
        &self,
        resolved_source: &str,
        on_missing: Option<&ResolveMissingCallback>,
        install: impl Future<Output = Result<(), PackageManagerError>>,
    ) -> Result<bool, PackageManagerError> {
        if self.offline() {
            return Ok(false);
        }
        match on_missing {
            None => {
                install.await?;
                Ok(true)
            }
            Some(on_missing) => match on_missing(resolved_source)? {
                MissingSourceAction::Skip => Ok(false),
                MissingSourceAction::Error => Err(PackageManagerError(format!(
                    "Missing source: {resolved_source}"
                ))),
                MissingSourceAction::Install => {
                    install.await?;
                    Ok(true)
                }
            },
        }
    }

    /// The user entry a project `autoload: false` delta resolves over,
    /// upstream's `findAutoloadDeltaBase`.
    fn find_autoload_delta_base(
        &self,
        pkg: &PackageSourceView,
        scope: SourceScope,
        sources: &[ScopedSource],
    ) -> Option<(String, SourceScope)> {
        if scope != SourceScope::Project {
            return None;
        }
        let filter = pkg.filter()?;
        if filter.autoload != Some(false) {
            return None;
        }
        let identity = self.get_package_identity(&filter.source, Some(scope));
        sources
            .iter()
            .find(|entry| {
                entry.scope == SourceScope::User
                    && self.get_package_identity(entry.pkg.source(), Some(SourceScope::User))
                        == identity
            })
            .map(|entry| (entry.pkg.source().to_string(), SourceScope::User))
    }

    /// Resolve a local source to its resources, upstream's
    /// `resolveLocalExtensionSource`: a file is one extension, a directory
    /// collects or falls back to itself as one extension.
    fn resolve_local_extension_source(
        &self,
        source: &LocalSource,
        accumulator: &mut ResourceAccumulator,
        filter: Option<&PackageFilter>,
        metadata: &PathMetadata,
        base_dir: &str,
    ) {
        let resolved = self.resolve_path_from_base(&source.path, base_dir);
        let resolved_path = PathBuf::from(&resolved);
        if !resolved_path.exists() {
            return;
        }

        let Ok(stats) = std::fs::metadata(&resolved_path) else {
            return;
        };
        if stats.is_file() {
            let mut metadata = metadata.clone();
            metadata.base_dir = resolved_path
                .parent()
                .map(|parent| parent.to_string_lossy().into_owned());
            add_resource(
                get_target_map(accumulator, ResourceType::Extensions),
                &resolved_path,
                &metadata,
                true,
            );
            return;
        }
        if stats.is_dir() {
            let mut metadata = metadata.clone();
            metadata.base_dir = Some(resolved);
            if !self.collect_package_resources(&resolved_path, accumulator, filter, &metadata) {
                add_resource(
                    get_target_map(accumulator, ResourceType::Extensions),
                    &resolved_path,
                    &metadata,
                    true,
                );
            }
        }
    }

    /// Install a parsed source for resolve, upstream's `installParsedSource`.
    async fn install_parsed_source(
        &self,
        parsed: &ParsedSource,
        scope: SourceScope,
    ) -> Result<(), PackageManagerError> {
        match parsed {
            ParsedSource::Crate(crate_source) => {
                self.install_crate(crate_source, scope, scope == SourceScope::Temporary)
                    .await
            }
            ParsedSource::Git(git_source) => self.install_git(git_source, scope).await,
            ParsedSource::Tarball(tarball) => {
                self.install_tarball(tarball, scope, scope == SourceScope::Temporary)
                    .await
            }
            ParsedSource::Local(_) => Ok(()),
        }
    }

    /// List the configured packages, upstream's `listConfiguredPackages`:
    /// user scope first, project second, install paths resolved when they
    /// exist.
    pub fn list_configured_packages(&self) -> Vec<ConfiguredPackage> {
        let manager = self.settings_lock();
        let global_settings = manager.get_global_settings();
        let project_settings = manager.get_project_settings();
        drop(manager);
        let mut configured_packages: Vec<ConfiguredPackage> = Vec::new();

        for pkg in project_packages(&global_settings) {
            let source = pkg.source().to_string();
            configured_packages.push(ConfiguredPackage {
                source: source.clone(),
                scope: SourceScope::User,
                filtered: pkg.filter().is_some(),
                installed_path: self
                    .get_installed_path(&source, SourceScope::User)
                    .ok()
                    .flatten(),
            });
        }

        for pkg in project_packages(&project_settings) {
            let source = pkg.source().to_string();
            configured_packages.push(ConfiguredPackage {
                source: source.clone(),
                scope: SourceScope::Project,
                filtered: pkg.filter().is_some(),
                installed_path: self
                    .get_installed_path(&source, SourceScope::Project)
                    .ok()
                    .flatten(),
            });
        }

        configured_packages
    }

    /// The install path of a configured source, upstream's
    /// `getInstalledPath`: the path when it exists on disk.
    ///
    /// # Errors
    /// A parse failure of the source.
    pub fn get_installed_path(
        &self,
        source: &str,
        scope: SourceScope,
    ) -> Result<Option<String>, PackageManagerError> {
        let path = match self.parse_source(source)? {
            ParsedSource::Crate(parsed) => self.get_crate_install_path(&parsed, scope)?,
            ParsedSource::Git(parsed) => self.get_git_install_path(&parsed, scope)?,
            ParsedSource::Tarball(parsed) => self.get_tarball_install_path(&parsed, scope)?,
            ParsedSource::Local(parsed) => {
                let base_dir = self.get_base_dir_for_scope(scope)?;
                self.resolve_path_from_base(&parsed.path, &base_dir)
            }
        };
        Ok(Path::new(&path).exists().then_some(path))
    }
}

/// The settings `packages` entries parsed, upstream's
/// `settings.packages ?? []` reads.
fn project_packages(
    settings: &serde_json::Map<String, serde_json::Value>,
) -> Vec<PackageSourceView> {
    settings
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .map(|entries| entries.iter().map(PackageSourceView::parse).collect())
        .unwrap_or_default()
}

/// The settings strings for one resource type, upstream's
/// `settings[resourceType] ?? []`.
fn settings_strings(
    settings: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Vec<String> {
    settings
        .get(key)
        .and_then(serde_json::Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

// =============================================================================
// Install, remove, update
// =============================================================================

impl<S: SettingsStorage + 'static> DefaultPackageManager<S> {
    /// Install a source without persisting it, upstream's `install`.
    ///
    /// # Errors
    /// A parse failure, the project-trust gate, or the channel's install
    /// failure.
    pub async fn install(&self, source: &str, local: bool) -> Result<(), PackageManagerError> {
        let parsed = self.parse_source(source)?;
        let scope = if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        self.assert_project_trusted_for_scope(scope)?;
        self.with_progress(
            ProgressAction::Install,
            source,
            &format!("Installing {source}..."),
            async move {
                match &parsed {
                    ParsedSource::Crate(crate_source) => {
                        self.install_crate(crate_source, scope, false).await
                    }
                    ParsedSource::Git(git_source) => self.install_git(git_source, scope).await,
                    ParsedSource::Tarball(tarball) => {
                        self.install_tarball(tarball, scope, false).await
                    }
                    ParsedSource::Local(local_source) => {
                        let resolved = self.resolve_path(&local_source.path);
                        if !Path::new(&resolved).exists() {
                            return Err(PackageManagerError(format!(
                                "Path does not exist: {resolved}"
                            )));
                        }
                        Ok(())
                    }
                }
            },
        )
        .await
    }

    /// Install and persist the source, upstream's `installAndPersist`.
    ///
    /// # Errors
    /// The install's errors, or the settings write failure.
    pub async fn install_and_persist(
        &self,
        source: &str,
        local: bool,
    ) -> Result<(), PackageManagerError> {
        self.install(source, local).await?;
        self.add_source_to_settings(source, local)?;
        Ok(())
    }

    /// Remove a source's installed artifacts, upstream's `remove`.
    ///
    /// # Errors
    /// A parse failure, the project-trust gate, or the channel's removal
    /// failure.
    pub async fn remove(&self, source: &str, local: bool) -> Result<(), PackageManagerError> {
        let parsed = self.parse_source(source)?;
        let scope = if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        self.assert_project_trusted_for_scope(scope)?;
        self.with_progress(
            ProgressAction::Remove,
            source,
            &format!("Removing {source}..."),
            async move {
                match &parsed {
                    ParsedSource::Crate(crate_source) => {
                        let target_dir = self.get_crate_install_path(crate_source, scope)?;
                        remove_dir_all_force(Path::new(&target_dir));
                        Ok(())
                    }
                    ParsedSource::Git(git_source) => self.remove_git(git_source, scope),
                    ParsedSource::Tarball(tarball) => {
                        let target_dir = self.get_tarball_install_path(tarball, scope)?;
                        remove_dir_all_force(Path::new(&target_dir));
                        Ok(())
                    }
                    ParsedSource::Local(_) => Ok(()),
                }
            },
        )
        .await
    }

    /// Remove and persist, upstream's `removeAndPersist`.
    ///
    /// # Errors
    /// The removal's errors, or the settings write failure.
    pub async fn remove_and_persist(
        &self,
        source: &str,
        local: bool,
    ) -> Result<bool, PackageManagerError> {
        self.remove(source, local).await?;
        self.remove_source_from_settings(source, local)
    }

    /// Update one source or every configured source, upstream's `update`.
    ///
    /// # Errors
    /// A parse failure, or the no-matching-package message.
    pub async fn update(&self, source: Option<&str>) -> Result<(), PackageManagerError> {
        let (global_settings, project_settings) = {
            let manager = self.settings_lock();
            (
                manager.get_global_settings(),
                manager.get_project_settings(),
            )
        };
        let identity = source.map(|source| self.get_package_identity(source, None));
        let mut matched = false;
        let mut update_sources: Vec<ConfiguredUpdateSource> = Vec::new();

        for pkg in project_packages(&global_settings) {
            let source_str = pkg.source().to_string();
            if identity.as_ref().is_some_and(|identity| {
                *identity != self.get_package_identity(&source_str, Some(SourceScope::User))
            }) {
                continue;
            }
            matched = true;
            update_sources.push(ConfiguredUpdateSource::new(source_str, SourceScope::User));
        }
        for pkg in project_packages(&project_settings) {
            let source_str = pkg.source().to_string();
            if identity.as_ref().is_some_and(|identity| {
                *identity != self.get_package_identity(&source_str, Some(SourceScope::Project))
            }) {
                continue;
            }
            matched = true;
            update_sources.push(ConfiguredUpdateSource::new(
                source_str,
                SourceScope::Project,
            ));
        }

        if let Some(source) = source
            && !matched
        {
            let mut configured: Vec<PackageSourceView> = project_packages(&global_settings);
            configured.extend(project_packages(&project_settings));
            return Err(PackageManagerError(
                self.build_no_matching_package_message(source, &configured),
            ));
        }

        self.update_configured_sources(update_sources).await
    }

    /// The update dispatch, upstream's `updateConfiguredSources`: unpinned
    /// `crate:` sources check versions, git sources reconcile; both pools
    /// run concurrently.
    async fn update_configured_sources(
        &self,
        sources: Vec<ConfiguredUpdateSource>,
    ) -> Result<(), PackageManagerError> {
        if self.offline() || sources.is_empty() {
            return Ok(());
        }

        let mut crate_candidates: Vec<ConfiguredUpdateSource> = Vec::new();
        let mut git_candidates: Vec<ConfiguredUpdateSource> = Vec::new();
        for entry in sources {
            match self.parse_source(&entry.source)? {
                // Pinned crate versions are fixed; pinned git refs are
                // configured checkout targets, so they still reconcile when
                // the configured ref changes.
                ParsedSource::Crate(parsed) if !parsed.pinned => {
                    crate_candidates.push(ConfiguredUpdateSource::new(entry.source, entry.scope));
                }
                ParsedSource::Git(parsed) => git_candidates.push(ConfiguredUpdateSource {
                    source: entry.source,
                    scope: entry.scope,
                    git: Some(parsed),
                }),
                _ => {}
            }
        }

        let mut user_crate_updates: Vec<ConfiguredUpdateSource> = Vec::new();
        let mut project_crate_updates: Vec<ConfiguredUpdateSource> = Vec::new();
        let checks: Vec<_> = crate_candidates
            .into_iter()
            .map(|entry| {
                let this = &self;
                async move {
                    let should_update = this.should_update_crate_source(&entry).await;
                    (entry, should_update)
                }
            })
            .collect();
        let check_results = run_with_concurrency(checks, UPDATE_CHECK_CONCURRENCY).await;
        for (entry, should_update) in check_results {
            if !should_update {
                continue;
            }
            if entry.scope == SourceScope::User {
                user_crate_updates.push(entry);
            } else {
                project_crate_updates.push(entry);
            }
        }

        let mut tasks: Vec<UpdateTask<'_>> = Vec::new();
        if !user_crate_updates.is_empty() {
            tasks.push(Box::pin(
                self.update_crate_batch(user_crate_updates, SourceScope::User),
            ));
        }
        if !project_crate_updates.is_empty() {
            tasks.push(Box::pin(
                self.update_crate_batch(project_crate_updates, SourceScope::Project),
            ));
        }
        if !git_candidates.is_empty() {
            let git_tasks: Vec<_> = git_candidates
                .into_iter()
                .map(|entry| {
                    let this = &self;
                    async move {
                        this.with_progress(
                            ProgressAction::Update,
                            &entry.source,
                            &format!("Updating {}...", entry.source),
                            async {
                                match entry.git.clone() {
                                    Some(git) => this.update_git(&git, entry.scope).await,
                                    None => Ok(()),
                                }
                            },
                        )
                        .await
                    }
                })
                .collect();
            tasks.push(Box::pin(async move {
                run_with_concurrency(git_tasks, GIT_UPDATE_CONCURRENCY).await;
                Ok(())
            }));
        }

        for task in tasks {
            task.await?;
        }
        Ok(())
    }

    /// Whether a configured crate source needs its install refreshed,
    /// upstream's `shouldUpdateNpmSource`: a missing install updates, a
    /// lookup failure preserves the update behavior (upstream's catch),
    /// otherwise the registry's latest must exceed the installed version.
    async fn should_update_crate_source(&self, entry: &ConfiguredUpdateSource) -> bool {
        let Ok(ParsedSource::Crate(parsed)) = self.parse_source(&entry.source) else {
            return true;
        };
        let Ok(installed_path) = self.get_crate_install_path(&parsed, entry.scope) else {
            return true;
        };
        let installed_version = if Path::new(&installed_path).exists() {
            Self::get_installed_crate_version(&installed_path)
        } else {
            None
        };
        let Some(installed_version) = installed_version else {
            return true;
        };

        self.get_latest_crate_version(&parsed.name)
            .await
            .map_or(true, |target_version| {
                semver::Version::parse(&target_version).is_ok_and(|target| {
                    semver::Version::parse(&installed_version)
                        .is_ok_and(|installed| target > installed)
                })
            })
    }

    /// Update a scope's crate sources in one progress span, upstream's
    /// `updateNpmBatch` — the single npm invocation restates to sequential
    /// cargo installs, one per package.
    async fn update_crate_batch(
        &self,
        sources: Vec<ConfiguredUpdateSource>,
        scope: SourceScope,
    ) -> Result<(), PackageManagerError> {
        if sources.is_empty() {
            return Ok(());
        }

        let source_label = if sources.len() == 1 {
            sources[0].source.clone()
        } else {
            match scope {
                SourceScope::Project => "project crate packages".to_string(),
                _ => "user crate packages".to_string(),
            }
        };
        let message = if sources.len() == 1 {
            format!("Updating {}...", sources[0].source)
        } else {
            match scope {
                SourceScope::Project => "Updating project crate packages...".to_string(),
                _ => "Updating user crate packages...".to_string(),
            }
        };

        self.with_progress(ProgressAction::Update, &source_label, &message, async {
            for entry in sources {
                let ParsedSource::Crate(parsed) = self.parse_source(&entry.source)? else {
                    continue;
                };
                self.install_crate(&parsed, scope, false).await?;
            }
            Ok(())
        })
        .await
    }

    /// Check the configured sources for available updates, upstream's
    /// `checkForAvailableUpdates`.
    ///
    /// # Errors
    /// Currently never fails: a source that does not parse or resolve
    /// skips, so per-source lookup failures degrade to no update reported.
    pub async fn check_for_available_updates(
        &self,
    ) -> Result<Vec<PackageUpdate>, PackageManagerError> {
        if self.offline() {
            return Ok(Vec::new());
        }

        let (global_settings, project_settings) = {
            let manager = self.settings_lock();
            (
                manager.get_global_settings(),
                manager.get_project_settings(),
            )
        };
        let mut all_packages: Vec<ScopedSource> = Vec::new();
        for pkg in project_packages(&project_settings) {
            all_packages.push(ScopedSource {
                pkg,
                scope: SourceScope::Project,
            });
        }
        for pkg in project_packages(&global_settings) {
            all_packages.push(ScopedSource {
                pkg,
                scope: SourceScope::User,
            });
        }

        let package_sources = self.dedupe_packages(&all_packages);
        let checks: Vec<_> = package_sources
            .into_iter()
            .filter(|entry| entry.scope != SourceScope::Temporary)
            .map(|entry| {
                let this = &self;
                async move {
                    let source = entry.pkg.source().to_string();
                    let parsed = this.parse_source(&source).ok()?;
                    match parsed {
                        ParsedSource::Local(_) | ParsedSource::Tarball(_) => None,
                        ParsedSource::Crate(parsed) if parsed.pinned => None,
                        ParsedSource::Crate(parsed) => {
                            let installed_path =
                                this.get_crate_install_path(&parsed, entry.scope).ok()?;
                            if !Path::new(&installed_path).exists() {
                                return None;
                            }
                            let installed_version =
                                Self::get_installed_crate_version(&installed_path)?;
                            let target_version =
                                this.get_latest_crate_version(&parsed.name).await.ok()?;
                            let has_update = semver::Version::parse(&target_version)
                                .ok()
                                .and_then(|target| {
                                    semver::Version::parse(&installed_version)
                                        .ok()
                                        .map(|installed| target > installed)
                                })
                                .unwrap_or(false);
                            has_update.then_some(PackageUpdate {
                                source,
                                display_name: parsed.name,
                                update_type: PackageUpdateType::Crate,
                                scope: entry.scope,
                            })
                        }
                        ParsedSource::Git(parsed) => {
                            let installed_path =
                                this.get_git_install_path(&parsed, entry.scope).ok()?;
                            if !Path::new(&installed_path).exists() {
                                return None;
                            }
                            let has_update = this.git_has_available_update(&installed_path).await;
                            has_update.then(|| PackageUpdate {
                                source,
                                display_name: format!("{}/{}", parsed.host, parsed.path),
                                update_type: PackageUpdateType::Git,
                                scope: entry.scope,
                            })
                        }
                    }
                }
            })
            .collect();
        let results = run_with_concurrency(checks, UPDATE_CHECK_CONCURRENCY).await;
        Ok(results.into_iter().flatten().collect())
    }

    /// The crates.io latest-version lookup, upstream's `getLatestNpmVersion`
    /// restated to the registry API: one request answers `max_version`, the
    /// version-array machinery dropping with npm.
    ///
    /// # Errors
    /// The request failure, an empty response, or an unexpected body.
    async fn get_latest_crate_version(&self, name: &str) -> Result<String, PackageManagerError> {
        let url = format!("https://crates.io/api/v1/crates/{name}");
        let signal = CancellationToken::new();
        let response = fetch_with_retry(
            &self.http_client,
            &url,
            vec![
                (
                    "User-Agent".to_string(),
                    get_pi_user_agent(crate::config::VERSION),
                ),
                ("accept".to_string(), "application/json".to_string()),
            ],
            signal,
            FetchRetryOptions {
                timeout_ms: Some(NETWORK_TIMEOUT_MS),
                ..FetchRetryOptions::default()
            },
        )
        .await
        .map_err(|error| PackageManagerError(error.to_string()))?;
        if response.status < 200 || response.status >= 300 {
            return Err(PackageManagerError(format!(
                "Unexpected response from crates.io: HTTP {}",
                response.status
            )));
        }
        let text = pi_ai::http::client::read_body_text(response.body)
            .await
            .map_err(|error| PackageManagerError(error.to_string()))?;
        let raw = text.trim();
        if raw.is_empty() {
            return Err(PackageManagerError(
                "Empty response from crates.io".to_string(),
            ));
        }
        let parsed: serde_json::Value = serde_json::from_str(raw)
            .map_err(|_| PackageManagerError("Unexpected response from crates.io".to_string()))?;
        parsed
            .get("crate")
            .and_then(|crate_field| crate_field.get("max_version"))
            .and_then(serde_json::Value::as_str)
            .filter(|version| !version.is_empty())
            .map(str::to_string)
            .ok_or_else(|| PackageManagerError("Unexpected response from crates.io".to_string()))
    }
}

/// A configured source collected for the update pass, upstream's
/// `ConfiguredUpdateSource` with its parsed git source riding alongside.
struct ConfiguredUpdateSource {
    source: String,
    scope: SourceScope,
    git: Option<GitSource>,
}

impl ConfiguredUpdateSource {
    const fn new(source: String, scope: SourceScope) -> Self {
        Self {
            source,
            scope,
            git: None,
        }
    }
}

/// One queued update future, the pinned boxed shape the update pools drive.
type UpdateTask<'a> =
    std::pin::Pin<Box<dyn Future<Output = Result<(), PackageManagerError>> + Send + 'a>>;

/// One indexed worker future, the shape `run_with_concurrency` tracks so
/// results keep task order.
type IndexedTask<'a, T> = std::pin::Pin<Box<dyn Future<Output = (usize, T)> + Send + 'a>>;

/// Run futures over a bounded worker pool, upstream's `runWithConcurrency`:
/// `limit` workers pull from a queue, results keep task order. The workers
/// drive inline (no spawned tasks), so the futures may borrow `self`.
#[expect(
    clippy::expect_used,
    reason = "every queued task's slot is filled before the collect; a None slot is an internal scheduling-invariant break, not a runtime condition"
)]
async fn run_with_concurrency<'a, T, F>(tasks: Vec<F>, limit: usize) -> Vec<T>
where
    F: Future<Output = T> + Send + 'a,
{
    if tasks.is_empty() {
        return Vec::new();
    }
    let worker_limit = limit.max(1);
    let total = tasks.len();
    let mut queue = tasks.into_iter().enumerate();
    let mut in_flight: futures_util::stream::FuturesUnordered<IndexedTask<'a, T>> =
        futures_util::stream::FuturesUnordered::new();
    let mut results: Vec<Option<T>> = (0..total).map(|_| None).collect();

    loop {
        while in_flight.len() < worker_limit {
            let Some((index, task)) = queue.next() else {
                break;
            };
            in_flight.push(Box::pin(async move {
                let value = task.await;
                (index, value)
            }));
        }
        if in_flight.is_empty() {
            break;
        }
        if let Some((index, value)) = futures_util::StreamExt::next(&mut in_flight).await {
            results[index] = Some(value);
        }
    }

    results
        .into_iter()
        .map(|slot| {
            slot.expect(
                "every queued task completes; a None slot is a scheduling invariant break, not a runtime condition",
            )
        })
        .collect()
}

/// Force-remove a directory tree, upstream's `rmSync(..., { recursive: true,
/// force: true })`: failures swallow, matching node's `force`.
fn remove_dir_all_force(path: &Path) {
    let _ = std::fs::remove_dir_all(path);
    let _ = std::fs::remove_file(path);
}

// =============================================================================
// Install channels and paths
// =============================================================================

/// The install receipt, the provenance marker ADR 0007's spawn gate
/// re-verifies.
///
/// `pi-package-install.json` at the package root carries the channel,
/// source, resolved version/ref, and the per-file sha256 map (receipt file
/// excluded) the extension host fails closed on when any byte changes.
/// Local path packages carry no receipt — they resolve in place and their
/// trust flows through project trust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallReceipt {
    /// The channel that produced the install.
    pub channel: String,
    /// The original source string.
    pub source: String,
    /// The resolved crate version, the `crate:` channel only.
    pub resolved_version: Option<String>,
    /// The resolved git ref, the git channel only.
    pub resolved_ref: Option<String>,
    /// The per-file sha256 hex digests, relative posix paths.
    pub files: BTreeMap<String, String>,
}

/// The receipt's file name.
pub const INSTALL_RECEIPT_FILE: &str = "pi-package-install.json";

impl InstallReceipt {
    /// The receipt's wire shape, `kind`/`schemaVersion` mirroring the
    /// managed-install marker's vocabulary.
    #[must_use]
    pub fn to_value(&self) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        object.insert(
            "kind".to_string(),
            serde_json::Value::String("pi-package-install".to_string()),
        );
        object.insert("schemaVersion".to_string(), serde_json::Value::from(1));
        object.insert(
            "channel".to_string(),
            serde_json::Value::String(self.channel.clone()),
        );
        object.insert(
            "source".to_string(),
            serde_json::Value::String(self.source.clone()),
        );
        if let Some(version) = &self.resolved_version {
            object.insert(
                "resolvedVersion".to_string(),
                serde_json::Value::String(version.clone()),
            );
        }
        if let Some(reference) = &self.resolved_ref {
            object.insert(
                "resolvedRef".to_string(),
                serde_json::Value::String(reference.clone()),
            );
        }
        object.insert(
            "files".to_string(),
            serde_json::Value::Object(
                self.files
                    .iter()
                    .map(|(path, digest)| (path.clone(), serde_json::Value::String(digest.clone())))
                    .collect(),
            ),
        );
        serde_json::Value::Object(object)
    }

    /// Parse a receipt file's value, the shape the spawn gate reads.
    ///
    /// # Errors
    /// A missing, malformed, or foreign `kind` — the fail-closed branch.
    pub fn from_value(value: &serde_json::Value) -> Result<Self, PackageManagerError> {
        if value.get("kind").and_then(serde_json::Value::as_str) != Some("pi-package-install") {
            return Err(PackageManagerError(
                "Not a pi package install receipt".to_string(),
            ));
        }
        if value
            .get("schemaVersion")
            .and_then(serde_json::Value::as_i64)
            != Some(1)
        {
            return Err(PackageManagerError(
                "Unsupported install receipt schema".to_string(),
            ));
        }
        let channel = value
            .get("channel")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| PackageManagerError("Install receipt carries no channel".to_string()))?
            .to_string();
        let files = value
            .get("files")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| PackageManagerError("Install receipt carries no file map".to_string()))?
            .iter()
            .filter_map(|(path, digest)| {
                digest
                    .as_str()
                    .map(|digest| (path.clone(), digest.to_string()))
            })
            .collect();
        Ok(Self {
            channel,
            source: value
                .get("source")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            resolved_version: value
                .get("resolvedVersion")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            resolved_ref: value
                .get("resolvedRef")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            files,
        })
    }
}

/// Hash the regular files under a package root into the receipt's map,
/// relative posix paths, the receipt file itself excluded.
fn hash_package_files(package_root: &Path) -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let mut stack = vec![package_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(path);
                continue;
            }
            if !metadata.is_file() {
                continue;
            }
            let relative = path
                .strip_prefix(package_root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            if relative == INSTALL_RECEIPT_FILE {
                continue;
            }
            if let Ok(contents) = std::fs::read(&path) {
                let mut hasher = sha2::Sha256::new();
                hasher.update(&contents);
                files.insert(relative, format!("{:x}", hasher.finalize()));
            }
        }
    }
    files
}

/// Write the receipt into a freshly staged package root.
///
/// # Errors
/// The receipt's serialization failure, or the file write.
fn write_install_receipt(
    stage_root: &Path,
    receipt: &InstallReceipt,
) -> Result<(), PackageManagerError> {
    let receipt_path = stage_root.join(INSTALL_RECEIPT_FILE);
    let value = receipt.to_value();
    let rendered = serde_json::to_string_pretty(&value)
        .map_err(|error| PackageManagerError(error.to_string()))?;
    std::fs::write(&receipt_path, format!("{rendered}\n"))
        .map_err(|error| PackageManagerError(error.to_string()))
}

impl<S: SettingsStorage + 'static> DefaultPackageManager<S> {
    /// The unique staging root one install builds into, the atomic-rename
    /// rule's staging half: the name carries the pid so concurrent installs
    /// never collide, and a leftover stage from a dead process is swept
    /// before the build starts.
    fn staging_root(install_root: &Path, name: &str) -> PathBuf {
        let suffix = std::process::id();
        let stage = install_root.join(format!(".{name}.staging-{suffix}"));
        remove_dir_all_force(&stage);
        stage
    }

    /// Install a `crate:` source, upstream's `installNpm` restated: cargo
    /// compiles into a staging root with `--locked`, and one atomic rename
    /// puts the package in place with its receipt.
    ///
    /// # Errors
    /// The trust gate, the cargo build failure, or the receipt write.
    async fn install_crate(
        &self,
        source: &CrateSource,
        scope: SourceScope,
        temporary: bool,
    ) -> Result<(), PackageManagerError> {
        let install_root = self.get_crate_install_root(scope, temporary)?;
        std::fs::create_dir_all(&install_root)
            .map_err(|error| PackageManagerError(error.to_string()))?;
        Self::ensure_git_ignore(&install_root)?;
        let target_dir = self.get_crate_install_path(source, scope)?;
        let stage_root = Self::staging_root(&install_root, &source.name);

        let mut args: Vec<String> = vec![
            "install".to_string(),
            "--locked".to_string(),
            "--root".to_string(),
            stage_root.to_string_lossy().into_owned(),
        ];
        if let Some(version) = &source.version {
            args.push("--version".to_string());
            args.push(version.clone());
        }
        args.push(source.name.clone());

        let run = async {
            let options = CommandRunOptions {
                cwd: Some(self.cwd.to_string_lossy().into_owned()),
                env: Vec::new(),
            };
            self.command_runner.run("cargo", &args, &options).await
        };
        let result: Result<(), PackageManagerError> = run.await;
        match result {
            Ok(()) => {}
            Err(error) => {
                remove_dir_all_force(&stage_root);
                return Err(error);
            }
        }

        // The installed tree replaces any previous install atomically: the
        // rename lands complete or not at all.
        let target_path = PathBuf::from(&target_dir);
        remove_dir_all_force(&target_path);
        std::fs::rename(&stage_root, &target_path).map_err(|error| {
            remove_dir_all_force(&stage_root);
            PackageManagerError(error.to_string())
        })?;

        let receipt = InstallReceipt {
            channel: "crate".to_string(),
            source: format!("crate:{}", source.name),
            resolved_version: source.version.clone(),
            resolved_ref: None,
            files: hash_package_files(&target_path),
        };
        write_install_receipt(&target_path, &receipt)
    }

    /// Install a tarball source, the prebuilt channel: the download streams
    /// under the size cap into a staging root, the unpack enforces the
    /// entry and path safety rules, and one atomic rename puts the package
    /// in place with its receipt.
    ///
    /// # Errors
    /// The trust gate, the download or unpack failure, a safety-rule
    /// rejection, or the receipt write.
    async fn install_tarball(
        &self,
        source: &TarballSource,
        scope: SourceScope,
        temporary: bool,
    ) -> Result<(), PackageManagerError> {
        let install_root = self.get_tarball_install_root(scope, temporary)?;
        std::fs::create_dir_all(&install_root)
            .map_err(|error| PackageManagerError(error.to_string()))?;
        Self::ensure_git_ignore(&install_root)?;
        let target_dir = self.get_tarball_install_path(source, scope)?;
        let target_path = PathBuf::from(&target_dir);
        let dir_name = target_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stage_root = Self::staging_root(&install_root, &dir_name);
        std::fs::create_dir_all(&stage_root)
            .map_err(|error| PackageManagerError(error.to_string()))?;

        let download = self.download_tarball(source);
        match download.await {
            Ok(archive_bytes) => match unpack_tarball(&archive_bytes, &stage_root) {
                Ok(()) => {}
                Err(error) => {
                    remove_dir_all_force(&stage_root);
                    return Err(error);
                }
            },
            Err(error) => {
                remove_dir_all_force(&stage_root);
                return Err(error);
            }
        }

        remove_dir_all_force(&target_path);
        std::fs::rename(&stage_root, &target_path).map_err(|error| {
            remove_dir_all_force(&stage_root);
            PackageManagerError(error.to_string())
        })?;

        let receipt = InstallReceipt {
            channel: "tarball".to_string(),
            source: source.url.clone(),
            resolved_version: None,
            resolved_ref: None,
            files: hash_package_files(&target_path),
        };
        write_install_receipt(&target_path, &receipt)
    }

    /// Stream the tarball download under [`MAX_TARBALL_BYTES`], upstream's
    /// npm-fetch restated over the [`HttpClient`] seam.
    ///
    /// # Errors
    /// The transport failure, a non-ok status, or the size cap.
    async fn download_tarball(
        &self,
        source: &TarballSource,
    ) -> Result<Vec<u8>, PackageManagerError> {
        let signal = CancellationToken::new();
        let response = fetch_with_retry(
            &self.http_client,
            &source.url,
            vec![
                (
                    "User-Agent".to_string(),
                    get_pi_user_agent(crate::config::VERSION),
                ),
                ("accept".to_string(), "application/octet-stream".to_string()),
            ],
            signal,
            FetchRetryOptions {
                timeout_ms: Some(NETWORK_TIMEOUT_MS),
                ..FetchRetryOptions::default()
            },
        )
        .await
        .map_err(|error| PackageManagerError(error.to_string()))?;
        if response.status < 200 || response.status >= 300 {
            return Err(PackageManagerError(format!(
                "Could not download package tarball from {}: HTTP {}",
                source.url, response.status
            )));
        }
        let mut body = response.body;
        let mut bytes: Vec<u8> = Vec::new();
        while let Some(chunk) = body
            .next_chunk()
            .await
            .map_err(|error| PackageManagerError(error.to_string()))?
        {
            if bytes.len() as u64 + chunk.len() as u64 > MAX_TARBALL_BYTES {
                return Err(PackageManagerError(format!(
                    "Package tarball from {} exceeds the {}-byte download cap",
                    source.url, MAX_TARBALL_BYTES
                )));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    /// Install a git source, upstream's `installGit` minus the npm
    /// dependency half: an existing checkout reconciles to its target, a
    /// new one clones and checks out, failures clean up.
    ///
    /// # Errors
    /// The trust gate, or the git failure.
    async fn install_git(
        &self,
        source: &GitSource,
        scope: SourceScope,
    ) -> Result<(), PackageManagerError> {
        let target_dir = self.get_git_install_path(source, scope)?;
        if Path::new(&target_dir).exists() {
            if let Some(reference) = &source.r#ref {
                self.ensure_git_ref(
                    &target_dir,
                    &["fetch".to_string(), "origin".to_string(), reference.clone()],
                    "FETCH_HEAD",
                )
                .await?;
                return Ok(());
            }
            let target = self.get_local_git_update_target(&target_dir).await?;
            self.ensure_git_ref(&target_dir, &target.fetch_args, &target.reference)
                .await?;
            return Ok(());
        }
        let git_root = self.get_git_install_root(scope)?;
        if let Some(git_root) = &git_root {
            Self::ensure_git_ignore(git_root)?;
        }
        if let Some(parent) = Path::new(&target_dir).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| PackageManagerError(error.to_string()))?;
        }
        remove_file_force(&Self::get_git_update_marker_path(Path::new(&target_dir)));

        let run = async {
            let options = CommandRunOptions {
                cwd: Some(self.cwd.to_string_lossy().into_owned()),
                env: Vec::new(),
            };
            self.command_runner
                .run(
                    "git",
                    &["clone".to_string(), source.repo.clone(), target_dir.clone()],
                    &options,
                )
                .await?;
            if let Some(reference) = &source.r#ref {
                let checkout_options = CommandRunOptions {
                    cwd: Some(target_dir.clone()),
                    env: Vec::new(),
                };
                self.command_runner
                    .run(
                        "git",
                        &["checkout".to_string(), reference.clone()],
                        &checkout_options,
                    )
                    .await?;
            }
            Ok(())
        };
        if let Err(error) = run.await {
            remove_dir_all_force(Path::new(&target_dir));
            Self::prune_empty_git_parents(Path::new(&target_dir), git_root.as_deref());
            return Err(error);
        }
        Ok(())
    }

    /// Update a git checkout to its target, upstream's `updateGit`.
    ///
    /// # Errors
    /// The git failure.
    async fn update_git(
        &self,
        source: &GitSource,
        scope: SourceScope,
    ) -> Result<(), PackageManagerError> {
        let target_dir = self.get_git_install_path(source, scope)?;
        if !Path::new(&target_dir).exists() {
            return self.install_git(source, scope).await;
        }

        if let Some(reference) = &source.r#ref {
            self.ensure_git_ref(
                &target_dir,
                &["fetch".to_string(), "origin".to_string(), reference.clone()],
                "FETCH_HEAD",
            )
            .await?;
            return Ok(());
        }

        let target = self.get_local_git_update_target(&target_dir).await?;
        self.ensure_git_ref(&target_dir, &target.fetch_args, &target.reference)
            .await
    }

    /// Whether the checkout's dependencies are missing, upstream's
    /// `hasMissingGitDependencies` restated to a no-op: the npm dependency
    /// half dropped with the npm channel, and a git checkout carries no
    /// package-manager dependencies to repair.
    const fn repair_missing_git_dependencies(_target_dir: &str) {}

    /// Clean untracked files out of the checkout, upstream's
    /// `cleanAndInstallGitDependencies` minus its npm half: extensions
    /// should be pristine, and a clean failure after nothing was deleted
    /// still surfaces.
    async fn clean_and_install_git_dependencies(
        &self,
        target_dir: &str,
        marker_path: &Path,
    ) -> Result<(), PackageManagerError> {
        let options = CommandRunOptions {
            cwd: Some(target_dir.to_string()),
            env: Vec::new(),
        };
        self.command_runner
            .run("git", &["clean".to_string(), "-fdx".to_string()], &options)
            .await?;
        remove_file_force(marker_path);
        Ok(())
    }

    /// Fetch and reset the checkout to the ref, upstream's `ensureGitRef`:
    /// only the ref we reset to fetches, the marker records an in-flight
    /// update, and a current checkout with a marker recovers its clean
    /// state.
    async fn ensure_git_ref(
        &self,
        target_dir: &str,
        fetch_args: &[String],
        reference: &str,
    ) -> Result<(), PackageManagerError> {
        let options = CommandRunOptions {
            cwd: Some(target_dir.to_string()),
            env: Vec::new(),
        };
        self.command_runner.run("git", fetch_args, &options).await?;

        let capture_options = CommandRunOptions {
            cwd: Some(target_dir.to_string()),
            env: Vec::new(),
        };
        let local_head = self
            .command_runner
            .run_capture(
                "git",
                &["rev-parse".to_string(), "HEAD".to_string()],
                &capture_options,
                Some(NETWORK_TIMEOUT_MS),
            )
            .await?;
        let mut commit_ref_args = vec!["rev-parse".to_string()];
        commit_ref_args.push(format!("{reference}^{{commit}}"));
        let target_head = self
            .command_runner
            .run_capture(
                "git",
                &commit_ref_args,
                &capture_options,
                Some(NETWORK_TIMEOUT_MS),
            )
            .await?;
        let marker_path = Self::get_git_update_marker_path(Path::new(target_dir));
        if local_head.trim() == target_head.trim() {
            if marker_path.exists() {
                self.clean_and_install_git_dependencies(target_dir, &marker_path)
                    .await?;
            } else {
                Self::repair_missing_git_dependencies(target_dir);
            }
            return Ok(());
        }

        std::fs::write(&marker_path, "").map_err(|error| PackageManagerError(error.to_string()))?;
        let mut reset_args = vec!["reset".to_string(), "--hard".to_string()];
        reset_args.push(format!("{reference}^{{commit}}"));
        self.command_runner
            .run("git", &reset_args, &options)
            .await?;
        self.clean_and_install_git_dependencies(target_dir, &marker_path)
            .await
    }

    /// Refresh a temporary git checkout, upstream's
    /// `refreshTemporaryGitSource`: a failed refresh keeps the cached
    /// checkout.
    async fn refresh_temporary_git_source(&self, source: &GitSource, source_str: &str) {
        if self.offline() {
            return;
        }
        let update = self.update_git(source, SourceScope::Temporary);
        let result = self
            .with_progress(
                ProgressAction::Pull,
                source_str,
                &format!("Refreshing {source_str}..."),
                update,
            )
            .await;
        // Keep cached temporary checkout if refresh fails.
        let _ = result;
    }

    /// Remove a git checkout and its marker, upstream's `removeGit`.
    ///
    /// # Errors
    /// The trust gate (project scope).
    fn remove_git(
        &self,
        source: &GitSource,
        scope: SourceScope,
    ) -> Result<(), PackageManagerError> {
        let target_dir = self.get_git_install_path(source, scope)?;
        remove_dir_all_force(Path::new(&target_dir));
        remove_file_force(&Self::get_git_update_marker_path(Path::new(&target_dir)));
        let install_root = self.get_git_install_root(scope)?;
        Self::prune_empty_git_parents(Path::new(&target_dir), install_root.as_deref());
        Ok(())
    }

    /// Remove empty parent directories up to the install root, upstream's
    /// `pruneEmptyGitParents`.
    fn prune_empty_git_parents(target_dir: &Path, install_root: Option<&Path>) {
        let Some(install_root) = install_root else {
            return;
        };
        let resolved_root =
            std::fs::canonicalize(install_root).unwrap_or_else(|_| install_root.to_path_buf());
        let mut current = match target_dir.parent() {
            Some(parent) => parent.to_path_buf(),
            None => return,
        };
        loop {
            if !(current.starts_with(&resolved_root) && current != resolved_root) {
                break;
            }
            if !current.exists() {
                current = match current.parent() {
                    Some(parent) => parent.to_path_buf(),
                    None => break,
                };
                continue;
            }
            let Ok(mut entries) = std::fs::read_dir(&current) else {
                break;
            };
            if entries.next().is_some() {
                break;
            }
            if std::fs::remove_dir(&current).is_err() {
                break;
            }
            current = match current.parent() {
                Some(parent) => parent.to_path_buf(),
                None => break,
            };
        }
    }

    /// Prepare an install root for a managed channel, upstream's
    /// `ensureNpmProject` minus the package.json half: the cloud-sync
    /// marker and the gitignore keep the manager's trees out of sync
    /// clients and git status.
    ///
    /// # Errors
    /// The directory creation or gitignore write failure.
    fn ensure_git_ignore(dir: &Path) -> Result<(), PackageManagerError> {
        if !dir.exists() {
            std::fs::create_dir_all(dir).map_err(|error| PackageManagerError(error.to_string()))?;
        }
        crate::utils::paths::mark_path_ignored_by_cloud_sync(&dir.to_string_lossy());
        let ignore_path = dir.join(".gitignore");
        if !ignore_path.exists() {
            std::fs::write(&ignore_path, "*\n!.gitignore\n")
                .map_err(|error| PackageManagerError(error.to_string()))?;
        }
        Ok(())
    }

    /// The `crate:` channel's install root, upstream's `getNpmInstallRoot`
    /// restated: user → `<agentDir>/crates`, project →
    /// `<cwd>/.pi/crates` (trust-gated), temporary → the managed temp dir.
    fn get_crate_install_root(
        &self,
        scope: SourceScope,
        temporary: bool,
    ) -> Result<PathBuf, PackageManagerError> {
        if temporary {
            return self.get_temporary_dir("crate", None);
        }
        match scope {
            SourceScope::Project => {
                self.assert_project_trusted_for_scope(scope)?;
                Ok(self.cwd.join(CONFIG_DIR_NAME).join("crates"))
            }
            SourceScope::User => Ok(self.agent_dir.join("crates")),
            SourceScope::Temporary => unreachable!("temporary handled above"),
        }
    }

    /// The `crate:` channel's package dir, upstream's `getManagedNpmInstallPath`
    /// restated: `<installRoot>/<name>`.
    /// The path computation the installed-path surface reports; public for the
    /// suite's traversal and layout assertions.
    ///
    /// # Errors
    /// The project-trust gate (project scope), or the managed-path
    /// resolution failure.
    pub fn get_crate_install_path(
        &self,
        source: &CrateSource,
        scope: SourceScope,
    ) -> Result<String, PackageManagerError> {
        if scope == SourceScope::Temporary {
            let dir = self.get_temporary_dir("crate", None)?;
            return Ok(self
                .resolve_managed_path(&dir, &source.name, &[])?
                .to_string_lossy()
                .into_owned());
        }
        if scope == SourceScope::Project {
            self.assert_project_trusted_for_scope(scope)?;
        }
        let install_root = self.get_crate_install_root(scope, false)?;
        Ok(self
            .resolve_managed_path(&install_root, &source.name, &[])?
            .to_string_lossy()
            .into_owned())
    }

    /// The tarball channel's install root: user → `<agentDir>/tarballs`,
    /// project → `<cwd>/.pi/tarballs` (trust-gated), temporary → the
    /// managed temp dir.
    fn get_tarball_install_root(
        &self,
        scope: SourceScope,
        temporary: bool,
    ) -> Result<PathBuf, PackageManagerError> {
        if temporary {
            return self.get_temporary_dir("tarball", None);
        }
        match scope {
            SourceScope::Project => {
                self.assert_project_trusted_for_scope(scope)?;
                Ok(self.cwd.join(CONFIG_DIR_NAME).join("tarballs"))
            }
            SourceScope::User => Ok(self.agent_dir.join("tarballs")),
            SourceScope::Temporary => unreachable!("temporary handled above"),
        }
    }

    /// The tarball channel's package dir: the sha256-8 of the URL names the
    /// directory, the only stable identity available before download.
    /// The path computation the installed-path surface reports; public for the
    /// suite's traversal and layout assertions.
    ///
    /// # Errors
    /// The project-trust gate (project scope), or the managed-path
    /// resolution failure.
    pub fn get_tarball_install_path(
        &self,
        source: &TarballSource,
        scope: SourceScope,
    ) -> Result<String, PackageManagerError> {
        if scope == SourceScope::Temporary {
            let dir = self.get_temporary_dir("tarball", None)?;
            return Ok(self
                .resolve_managed_path(&dir, &tarball_dir_name(&source.url), &[])?
                .to_string_lossy()
                .into_owned());
        }
        if scope == SourceScope::Project {
            self.assert_project_trusted_for_scope(scope)?;
        }
        let install_root = self.get_tarball_install_root(scope, false)?;
        Ok(self
            .resolve_managed_path(&install_root, &tarball_dir_name(&source.url), &[])?
            .to_string_lossy()
            .into_owned())
    }

    /// The git channel's install root, upstream's `getGitInstallRoot`:
    /// user → `<agentDir>/git`, project → `<cwd>/.pi/git` (trust-gated),
    /// temporary → `None`.
    fn get_git_install_root(
        &self,
        scope: SourceScope,
    ) -> Result<Option<PathBuf>, PackageManagerError> {
        if scope == SourceScope::Temporary {
            return Ok(None);
        }
        if scope == SourceScope::Project {
            self.assert_project_trusted_for_scope(scope)?;
            return Ok(Some(self.cwd.join(CONFIG_DIR_NAME).join("git")));
        }
        Ok(Some(self.agent_dir.join("git")))
    }

    /// The git channel's checkout path, upstream's `getGitInstallPath`:
    /// temporary sources hash under the temp folder, the rest join the
    /// install root's host/path tree.
    /// The path computation the installed-path surface reports; public for the
    /// suite's traversal and layout assertions.
    ///
    /// # Errors
    /// The managed-path resolution failure, or the missing git install
    /// root when no root applies to the scope.
    pub fn get_git_install_path(
        &self,
        source: &GitSource,
        scope: SourceScope,
    ) -> Result<String, PackageManagerError> {
        if scope == SourceScope::Temporary {
            let dir =
                self.get_temporary_dir(&format!("git-{}", source.host), Some(&source.path))?;
            return Ok(dir.to_string_lossy().into_owned());
        }
        let install_root = self
            .get_git_install_root(scope)?
            .ok_or_else(|| PackageManagerError("Missing git install root".to_string()))?;
        let segments: Vec<&str> = source.path.split('/').collect();
        Ok(self
            .resolve_managed_path(&install_root, &source.host, &segments)?
            .to_string_lossy()
            .into_owned())
    }

    /// The hash-named temporary dir, upstream's `getTemporaryDir`:
    /// `<tempRoot>/<prefix>/<sha256-8>` with the optional suffix joined.
    fn get_temporary_dir(
        &self,
        prefix: &str,
        suffix: Option<&str>,
    ) -> Result<PathBuf, PackageManagerError> {
        let temp_folder = get_extension_temp_folder(&self.agent_dir)
            .map_err(|error| PackageManagerError(error.to_string()))?;
        let root = self.resolve_managed_path(&temp_folder, prefix, &[])?;
        let hash = temporary_dir_hash(prefix, suffix.unwrap_or(""));
        let parts: Vec<&str> = suffix
            .filter(|suffix| !suffix.is_empty())
            .map_or_else(|| vec![hash.as_str()], |suffix| vec![hash.as_str(), suffix]);
        let first = parts.first().copied().unwrap_or("");
        let rest: Vec<&str> = parts.iter().skip(1).copied().collect();
        self.resolve_managed_path(&root, first, &rest)
    }

    /// Join under a managed root with the traversal gate, upstream's
    /// `resolveManagedPath`: the result must stay inside the root.
    #[expect(
        clippy::unused_self,
        reason = "the method mirrors upstream's PackageManager method shape and its seven call sites read as method dispatch"
    )]
    fn resolve_managed_path(
        &self,
        root: &Path,
        first: &str,
        rest: &[&str],
    ) -> Result<PathBuf, PackageManagerError> {
        let resolved_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        let mut resolved_path = resolved_root.clone();
        for segment in std::iter::once(first).chain(rest.iter().copied()) {
            for part in segment.split('/') {
                match part {
                    "" | "." => {}
                    ".." => {
                        resolved_path.pop();
                    }
                    other => resolved_path.push(other),
                }
            }
        }
        if resolved_path != resolved_root
            && !resolved_path.to_string_lossy().starts_with(&format!(
                "{}{}",
                resolved_root.to_string_lossy(),
                std::path::MAIN_SEPARATOR
            ))
        {
            return Err(PackageManagerError(format!(
                "Refusing to use path outside package install root: {}",
                resolved_path.to_string_lossy()
            )));
        }
        Ok(resolved_path)
    }

    /// The in-flight update marker, upstream's `getGitUpdateMarkerPath`.
    fn get_git_update_marker_path(target_dir: &Path) -> PathBuf {
        let basename = target_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        target_dir
            .parent()
            .unwrap_or(target_dir)
            .join(format!(".{basename}.pi-update-incomplete"))
    }

    /// The installed crate version, upstream's `getInstalledNpmVersion`
    /// restated to the receipt's `resolvedVersion` field with the
    /// `package.json` fallback: the receipt is the manager's own record,
    /// and a package.json version (the layout npm-era checkouts and
    /// hand-placed installs carry) answers the same question upstream read
    /// it for.
    fn get_installed_crate_version(installed_path: &str) -> Option<String> {
        if let Some(receipt) =
            std::fs::read_to_string(Path::new(installed_path).join(INSTALL_RECEIPT_FILE))
                .ok()
                .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
                .and_then(|value| InstallReceipt::from_value(&value).ok())
        {
            return receipt.resolved_version;
        }
        let package_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(Path::new(installed_path).join("package.json")).ok()?,
        )
        .ok()?;
        package_json
            .get("version")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    }
}

/// The sha256-8 the temporary dir hash carries, upstream's
/// `createHash("sha256")` truncated hex.
fn temporary_dir_hash(prefix: &str, suffix: &str) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(format!("{prefix}-{suffix}"));
    format!("{:x}", hasher.finalize())[..8].to_string()
}

/// The tarball install's directory name, the sha256-8 of the URL.
fn tarball_dir_name(url: &str) -> String {
    temporary_dir_hash("tarball", url)
}

/// Apply the entry's unix mode to the unpacked file, the tar mode the
/// extension binaries need to stay executable.
fn preserve_entry_mode(entry: &tar::Entry<'_, impl std::io::Read>, target: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = entry.header().mode().unwrap_or(0o644) & 0o777;
        let _ = std::fs::set_permissions(target, std::fs::Permissions::from_mode(mode));
    }
    #[cfg(not(unix))]
    {
        let _ = (entry, target);
    }
}

/// Force-remove a file, upstream's `rmSync(..., { force: true })`.
fn remove_file_force(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Unpack a gzip tarball under the safety rules: traversal and absolute
/// paths reject, escaping symlinks reject, special files reject, and the
/// entry-count and unpacked-size caps bound the extraction.
///
/// # Errors
/// The decompress failure, a safety-rule rejection, or a cap overrun.
pub fn unpack_tarball(archive_bytes: &[u8], stage_root: &Path) -> Result<(), PackageManagerError> {
    let gz = flate2::read::GzDecoder::new(archive_bytes);
    let mut archive = tar::Archive::new(gz);
    let mut total_bytes: u64 = 0;
    let mut entry_count: u64 = 0;

    for entry in archive
        .entries()
        .map_err(|error| PackageManagerError(error.to_string()))?
    {
        let mut entry = entry.map_err(|error| PackageManagerError(error.to_string()))?;
        entry_count += 1;
        if entry_count > MAX_TARBALL_ENTRIES {
            return Err(PackageManagerError(format!(
                "Package tarball carries more than {MAX_TARBALL_ENTRIES} entries"
            )));
        }
        let path_in_archive = entry
            .path()
            .map_err(|error| PackageManagerError(error.to_string()))?
            .to_path_buf();
        let path_str = path_in_archive.to_string_lossy().into_owned();
        let relative = path_str.trim_start_matches("./");
        if relative.is_empty()
            || Path::new(relative).is_absolute()
            || relative.split('/').any(|segment| segment == "..")
        {
            return Err(PackageManagerError(format!(
                "Package tarball entry escapes the staging root: {path_str}"
            )));
        }
        let target = stage_root.join(relative);
        let size = entry
            .header()
            .size()
            .map_err(|error| PackageManagerError(error.to_string()))?;
        total_bytes += size;
        if total_bytes > MAX_UNPACKED_BYTES {
            return Err(PackageManagerError(format!(
                "Package tarball unpack exceeds the {MAX_UNPACKED_BYTES}-byte cap"
            )));
        }
        match entry.header().entry_type() {
            tar::EntryType::Regular => {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|error| PackageManagerError(error.to_string()))?;
                }
                let mut contents = Vec::new();
                std::io::Read::read_to_end(&mut entry, &mut contents)
                    .map_err(|error| PackageManagerError(error.to_string()))?;
                if total_bytes + contents.len() as u64 > MAX_UNPACKED_BYTES {
                    return Err(PackageManagerError(format!(
                        "Package tarball unpack exceeds the {MAX_UNPACKED_BYTES}-byte cap"
                    )));
                }
                total_bytes += contents.len() as u64;
                std::fs::write(&target, &contents)
                    .map_err(|error| PackageManagerError(error.to_string()))?;
                preserve_entry_mode(&entry, &target);
            }
            tar::EntryType::Directory => {
                std::fs::create_dir_all(&target)
                    .map_err(|error| PackageManagerError(error.to_string()))?;
            }
            tar::EntryType::Symlink => {
                let link_target = entry
                    .link_name()
                    .map_err(|error| PackageManagerError(error.to_string()))?
                    .unwrap_or_default();
                let resolved = target.parent().unwrap_or(stage_root).join(&link_target);
                if !resolved.starts_with(stage_root) {
                    return Err(PackageManagerError(format!(
                        "Package tarball symlink escapes the staging root: {path_str}"
                    )));
                }
                #[cfg(unix)]
                std::os::unix::fs::symlink(&link_target, &target)
                    .map_err(|error| PackageManagerError(error.to_string()))?;
                #[cfg(not(unix))]
                std::fs::write(&target, &link_target.to_string_lossy().into_owned())
                    .map_err(|error| PackageManagerError(error.to_string()))?;
            }
            // Hard links and special files reject: a package tarball needs
            // none of them, and a hard link can escape the staging root.
            _ => {
                return Err(PackageManagerError(format!(
                    "Package tarball carries an unsupported entry type: {path_str}"
                )));
            }
        }
    }
    Ok(())
}

// =============================================================================
// Resource collection
// =============================================================================

/// The per-type accumulation maps, upstream's `ResourceAccumulator`.
///
/// The entries ride insertion order — upstream's JS `Map` iterates in
/// insertion order and the resolved output keeps it, so the Rust maps are
/// `IndexMap`s, not `HashMap`s.
#[derive(Debug, Default)]
pub struct ResourceAccumulator {
    /// Extension entries, upstream's `extensions` map.
    pub extensions: indexmap::IndexMap<PathBuf, AccumulatedResource>,
    /// Skill entries, upstream's `skills` map.
    pub skills: indexmap::IndexMap<PathBuf, AccumulatedResource>,
    /// Prompt entries, upstream's `prompts` map.
    pub prompts: indexmap::IndexMap<PathBuf, AccumulatedResource>,
    /// Theme entries, upstream's `themes` map.
    pub themes: indexmap::IndexMap<PathBuf, AccumulatedResource>,
}

/// One accumulated entry, upstream's `{ metadata, enabled }` map value.
#[derive(Debug, Clone)]
pub struct AccumulatedResource {
    /// The provenance, upstream's `metadata`.
    pub metadata: PathMetadata,
    /// Whether the resource loads, upstream's `enabled`.
    pub enabled: bool,
}

fn create_accumulator() -> ResourceAccumulator {
    ResourceAccumulator::default()
}

/// The target map for a resource type, upstream's `getTargetMap`.
const fn get_target_map(
    accumulator: &mut ResourceAccumulator,
    resource_type: ResourceType,
) -> &mut indexmap::IndexMap<PathBuf, AccumulatedResource> {
    match resource_type {
        ResourceType::Extensions => &mut accumulator.extensions,
        ResourceType::Skills => &mut accumulator.skills,
        ResourceType::Prompts => &mut accumulator.prompts,
        ResourceType::Themes => &mut accumulator.themes,
    }
}

/// Insert an entry when absent, upstream's `addResource`: the first
/// contributor wins name collisions.
fn add_resource(
    map: &mut indexmap::IndexMap<PathBuf, AccumulatedResource>,
    path: &Path,
    metadata: &PathMetadata,
    enabled: bool,
) {
    if path.as_os_str().is_empty() {
        return;
    }
    map.entry(path.to_path_buf())
        .or_insert_with(|| AccumulatedResource {
            metadata: metadata.clone(),
            enabled,
        });
}

/// Collect a package's resources, upstream's `collectPackageResources`:
/// filters apply per type, a `pi` manifest declares its entries, and the
/// convention directories are the fallback.
///
/// Returns whether any resource surfaced, upstream's `boolean` return the
/// local-directory fallback reads.
#[allow(
    clippy::too_many_lines,
    reason = "the upstream method's branches carry one-to-one"
)]
fn collect_package_resources_impl(
    package_root: &Path,
    accumulator: &mut ResourceAccumulator,
    filter: Option<&PackageFilter>,
    metadata: &PathMetadata,
) -> bool {
    if let Some(filter) = filter {
        let default_patterns: Vec<String> = Vec::new();
        for resource_type in ResourceType::all() {
            let patterns = filter_patterns(filter, resource_type);
            let target = get_target_map(accumulator, resource_type);
            if filter.autoload == Some(false) {
                apply_package_delta_filter(
                    package_root,
                    patterns.as_ref().unwrap_or(&default_patterns),
                    resource_type,
                    target,
                    metadata,
                );
            } else if let Some(patterns) = patterns {
                apply_package_filter(package_root, &patterns, resource_type, target, metadata);
            } else {
                collect_default_resources(package_root, resource_type, target, metadata);
            }
        }
        return true;
    }

    if let Some(manifest) = crate::pi_manifest::read_pi_manifest(&package_root.join("package.json"))
    {
        for resource_type in ResourceType::all() {
            let entries = manifest_entries(&manifest, resource_type);
            add_manifest_entries(
                entries,
                package_root,
                resource_type,
                get_target_map(accumulator, resource_type),
                metadata,
            );
        }
        return true;
    }

    let mut has_any_dir = false;
    for resource_type in ResourceType::all() {
        let dir = package_root.join(resource_type.as_str());
        if dir.exists() {
            let files = collect_resource_files(&dir, resource_type);
            for file in files {
                add_resource(
                    get_target_map(accumulator, resource_type),
                    &file,
                    metadata,
                    true,
                );
            }
            has_any_dir = true;
        }
    }
    if !has_any_dir
        && package_root.join("bin").is_dir()
        && package_root.join(INSTALL_RECEIPT_FILE).exists()
    {
        // The `crate:` channel's cargo-installed layout: no manifest and no
        // convention directories, just the compiled binaries under `bin/` —
        // the binaries are the package's extension entries. The receipt
        // gates the branch so an arbitrary local directory keeps upstream's
        // directory-itself behavior.
        let executables: Vec<PathBuf> = std::fs::read_dir(package_root.join("bin"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && is_executable(path))
            .collect();
        for file in executables {
            add_resource(
                get_target_map(accumulator, ResourceType::Extensions),
                &file,
                metadata,
                true,
            );
        }
        has_any_dir = true;
    }
    has_any_dir
}

/// The filter array one resource type carries, upstream's
/// `filter[resourceType]`.
fn filter_patterns(filter: &PackageFilter, resource_type: ResourceType) -> Option<Vec<String>> {
    match resource_type {
        ResourceType::Extensions => filter.extensions.clone(),
        ResourceType::Skills => filter.skills.clone(),
        ResourceType::Prompts => filter.prompts.clone(),
        ResourceType::Themes => filter.themes.clone(),
    }
}

/// The manifest array one resource type carries, upstream's
/// `manifest[resourceType]`.
fn manifest_entries(
    manifest: &crate::pi_manifest::PiManifest,
    resource_type: ResourceType,
) -> Option<Vec<String>> {
    match resource_type {
        ResourceType::Extensions => manifest.extensions.clone(),
        ResourceType::Skills => manifest.skills.clone(),
        ResourceType::Prompts => manifest.prompts.clone(),
        ResourceType::Themes => manifest.themes.clone(),
    }
}

/// The default-collection arm, upstream's `collectDefaultResources`: the
/// manifest wins, the convention directory is the fallback.
fn collect_default_resources(
    package_root: &Path,
    resource_type: ResourceType,
    target: &mut indexmap::IndexMap<PathBuf, AccumulatedResource>,
    metadata: &PathMetadata,
) {
    let manifest = crate::pi_manifest::read_pi_manifest(&package_root.join("package.json"));
    let entries = manifest
        .as_ref()
        .and_then(|manifest| manifest_entries(manifest, resource_type));
    if let Some(entries) = entries {
        add_manifest_entries(Some(entries), package_root, resource_type, target, metadata);
        return;
    }
    let dir = package_root.join(resource_type.as_str());
    if dir.exists() {
        let files = collect_resource_files(&dir, resource_type);
        for file in files {
            add_resource(target, &file, metadata, true);
        }
    }
}

/// Apply user patterns on top of the package's own collection, upstream's
/// `applyPackageFilter`: an empty array explicitly disables everything.
fn apply_package_filter(
    package_root: &Path,
    user_patterns: &[String],
    resource_type: ResourceType,
    target: &mut indexmap::IndexMap<PathBuf, AccumulatedResource>,
    metadata: &PathMetadata,
) {
    let (all_files, _) = collect_manifest_files(package_root, resource_type);

    if user_patterns.is_empty() {
        for file in &all_files {
            add_resource(target, file, metadata, false);
        }
        return;
    }

    let enabled_by_user = apply_patterns(&all_files, user_patterns, package_root);
    for file in &all_files {
        let enabled = enabled_by_user.contains(file);
        add_resource(target, file, metadata, enabled);
    }
}

/// Apply the autoload-disabled delta patterns, upstream's
/// `applyPackageDeltaFilter`: only the patterns' matches register.
fn apply_package_delta_filter(
    package_root: &Path,
    user_patterns: &[String],
    resource_type: ResourceType,
    target: &mut indexmap::IndexMap<PathBuf, AccumulatedResource>,
    metadata: &PathMetadata,
) {
    if user_patterns.is_empty() {
        return;
    }

    let (all_files, _) = collect_manifest_files(package_root, resource_type);
    let enabled_by_user = apply_autoload_disabled_patterns(&all_files, user_patterns, package_root);
    for (file_path, enabled) in enabled_by_user {
        add_resource(target, &file_path, metadata, enabled);
    }
}

/// Collect a package's files for one type, upstream's
/// `collectManifestFiles`: the manifest's non-override entries expand (globs
/// scan, exact entries may target dot paths), override patterns then select
/// the enabled set; the convention directory is the fallback.
fn collect_manifest_files(
    package_root: &Path,
    resource_type: ResourceType,
) -> (Vec<PathBuf>, HashSet<PathBuf>) {
    let manifest = crate::pi_manifest::read_pi_manifest(&package_root.join("package.json"));
    let entries = manifest
        .as_ref()
        .and_then(|manifest| manifest_entries(manifest, resource_type));
    if let Some(entries) = entries.filter(|entries| !entries.is_empty()) {
        let all_files = collect_files_from_manifest_entries(&entries, package_root, resource_type);
        let manifest_patterns: Vec<String> = entries
            .iter()
            .filter(|entry| is_override_pattern(entry))
            .cloned()
            .collect();
        let enabled_by_manifest = if manifest_patterns.is_empty() {
            all_files.iter().cloned().collect()
        } else {
            apply_patterns(&all_files, &manifest_patterns, package_root)
        };
        let enabled_files: Vec<PathBuf> = all_files
            .iter()
            .filter(|file| enabled_by_manifest.contains(*file))
            .cloned()
            .collect();
        return (enabled_files, enabled_by_manifest);
    }

    let convention_dir = package_root.join(resource_type.as_str());
    if !convention_dir.exists() {
        return (Vec::new(), HashSet::new());
    }
    let all_files = collect_resource_files(&convention_dir, resource_type);
    let enabled = all_files.iter().cloned().collect();
    (all_files, enabled)
}

/// Add the manifest entries that pass the override patterns, upstream's
/// `addManifestEntries`.
fn add_manifest_entries(
    entries: Option<Vec<String>>,
    root: &Path,
    resource_type: ResourceType,
    target: &mut indexmap::IndexMap<PathBuf, AccumulatedResource>,
    metadata: &PathMetadata,
) {
    let Some(entries) = entries else {
        return;
    };

    let all_files = collect_files_from_manifest_entries(&entries, root, resource_type);
    let patterns: Vec<String> = entries
        .iter()
        .filter(|entry| is_override_pattern(entry))
        .cloned()
        .collect();
    let enabled_paths = apply_patterns(&all_files, &patterns, root);

    for file in all_files {
        if enabled_paths.contains(&file) {
            add_resource(target, &file, metadata, true);
        }
    }
}

/// Expand the manifest entries into files, upstream's
/// `collectFilesFromManifestEntries`: non-override entries resolve — exact
/// entries resolve directly, glob entries scan under the root — then the
/// resolved paths collect their files.
fn collect_files_from_manifest_entries(
    entries: &[String],
    root: &Path,
    resource_type: ResourceType,
) -> Vec<PathBuf> {
    let source_entries: Vec<String> = entries
        .iter()
        .filter(|entry| !is_override_pattern(entry))
        .cloned()
        .collect();
    let resolved: Vec<PathBuf> = source_entries
        .iter()
        .flat_map(|entry| {
            if has_glob_pattern(entry) {
                expand_package_glob(entry, root)
                    .into_iter()
                    .map(PathBuf::from)
                    .collect::<Vec<PathBuf>>()
            } else {
                vec![normalize_manifest_entry(root, entry)]
            }
        })
        .collect();
    collect_files_from_paths(&resolved, resource_type)
}

/// Collect files from resolved paths, upstream's `collectFilesFromPaths`:
/// files contribute directly, directories recurse their resource walk.
fn collect_files_from_paths(paths: &[PathBuf], resource_type: ResourceType) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for path in paths {
        if !path.exists() {
            continue;
        }
        let Ok(stats) = std::fs::metadata(path) else {
            continue;
        };
        if stats.is_file() {
            files.push(path.clone());
        } else if stats.is_dir() {
            files.extend(collect_resource_files(path, resource_type));
        }
    }
    files
}

// =============================================================================
// Auto-discovery, output shaping, and the manager method wrappers
// =============================================================================

impl<S: SettingsStorage + 'static> DefaultPackageManager<S> {
    /// Collect a package's resources, upstream's `collectPackageResources`.
    #[expect(
        clippy::unused_self,
        reason = "the method mirrors upstream's PackageManager method shape; the five resolve-path call sites read as method dispatch"
    )]
    fn collect_package_resources(
        &self,
        package_root: &Path,
        accumulator: &mut ResourceAccumulator,
        filter: Option<&PackageFilter>,
        metadata: &PathMetadata,
    ) -> bool {
        collect_package_resources_impl(package_root, accumulator, filter, metadata)
    }

    /// The settings-array arm, upstream's `resolveLocalEntries`.
    fn resolve_local_entries(
        &self,
        entries: &[String],
        resource_type: ResourceType,
        target: &mut indexmap::IndexMap<PathBuf, AccumulatedResource>,
        metadata: &PathMetadata,
        base_dir: &str,
    ) {
        if entries.is_empty() {
            return;
        }

        let (plain, patterns) = split_patterns(entries);
        let resolved_plain: Vec<PathBuf> = plain
            .iter()
            .map(|entry| PathBuf::from(self.resolve_path_from_base(entry, base_dir)))
            .collect();
        let all_files = collect_files_from_paths(&resolved_plain, resource_type);
        let enabled_paths = apply_patterns(&all_files, &patterns, Path::new(base_dir));

        for file in all_files {
            let enabled = enabled_paths.contains(&file);
            add_resource(target, &file, metadata, enabled);
        }
    }

    /// The auto-discovery pass, upstream's `addAutoDiscoveredResources`:
    /// the agent-dir and `.pi` convention directories, the `.agents/skills`
    /// walk up to the git root, and the settings arrays' enable/disable
    /// overrides.
    #[allow(
        clippy::too_many_lines,
        reason = "the upstream method's four resource types carry one-to-one"
    )]
    fn add_auto_discovered_resources(
        &self,
        accumulator: &mut ResourceAccumulator,
        global_settings: &serde_json::Map<String, serde_json::Value>,
        project_settings: &serde_json::Map<String, serde_json::Value>,
        global_base_dir: &str,
        project_base_dir: &str,
    ) {
        let user_metadata = PathMetadata {
            source: "auto".to_string(),
            scope: SourceScope::User,
            origin: ResourceOrigin::TopLevel,
            base_dir: Some(global_base_dir.to_string()),
        };
        let project_metadata = PathMetadata {
            source: "auto".to_string(),
            scope: SourceScope::Project,
            origin: ResourceOrigin::TopLevel,
            base_dir: Some(project_base_dir.to_string()),
        };

        let user_overrides = (
            settings_strings(global_settings, "extensions"),
            settings_strings(global_settings, "skills"),
            settings_strings(global_settings, "prompts"),
            settings_strings(global_settings, "themes"),
        );
        let project_overrides = (
            settings_strings(project_settings, "extensions"),
            settings_strings(project_settings, "skills"),
            settings_strings(project_settings, "prompts"),
            settings_strings(project_settings, "themes"),
        );

        let user_dirs = (
            Path::new(global_base_dir).join("extensions"),
            Path::new(global_base_dir).join("skills"),
            Path::new(global_base_dir).join("prompts"),
            Path::new(global_base_dir).join("themes"),
        );
        let project_dirs = (
            Path::new(project_base_dir).join("extensions"),
            Path::new(project_base_dir).join("skills"),
            Path::new(project_base_dir).join("prompts"),
            Path::new(project_base_dir).join("themes"),
        );
        let user_agents_skills_dir = Path::new(&self.home()).join(".agents").join("skills");
        let project_trusted = self.settings_lock().is_project_trusted();
        let project_agents_skill_dirs: Vec<PathBuf> = if project_trusted {
            collect_ancestor_agents_skill_dirs(&self.cwd)
                .into_iter()
                .filter(|dir| {
                    std::fs::canonicalize(dir).ok()
                        != std::fs::canonicalize(&user_agents_skills_dir).ok()
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut add_resources = |resource_type: ResourceType,
                                 paths: Vec<PathBuf>,
                                 metadata: &PathMetadata,
                                 overrides: &[String],
                                 base_dir: &str| {
            let target = get_target_map(accumulator, resource_type);
            for path in paths {
                let enabled = is_enabled_by_overrides(&path, overrides, Path::new(base_dir));
                add_resource(target, &path, metadata, enabled);
            }
        };

        if project_trusted {
            // Project extensions from .pi/
            add_resources(
                ResourceType::Extensions,
                collect_auto_extension_entries(&project_dirs.0),
                &project_metadata,
                &project_overrides.0,
                project_base_dir,
            );

            // Project skills from .pi/
            add_resources(
                ResourceType::Skills,
                collect_auto_skill_entries(&project_dirs.1, SkillDiscoveryMode::Pi),
                &project_metadata,
                &project_overrides.1,
                project_base_dir,
            );
        }

        // Project skills from .agents/ (each with its own baseDir)
        for agents_skills_dir in project_agents_skill_dirs {
            let agents_base_dir = agents_skills_dir
                .parent()
                .map(Path::to_string_lossy)
                .map(Cow::into_owned)
                .unwrap_or_default();
            let agents_metadata = PathMetadata {
                base_dir: Some(agents_base_dir.clone()),
                ..project_metadata.clone()
            };
            add_resources(
                ResourceType::Skills,
                collect_auto_skill_entries(&agents_skills_dir, SkillDiscoveryMode::Agents),
                &agents_metadata,
                &project_overrides.1,
                &agents_base_dir,
            );
        }

        if project_trusted {
            add_resources(
                ResourceType::Prompts,
                collect_auto_prompt_entries(&project_dirs.2),
                &project_metadata,
                &project_overrides.2,
                project_base_dir,
            );
            add_resources(
                ResourceType::Themes,
                collect_auto_theme_entries(&project_dirs.3),
                &project_metadata,
                &project_overrides.3,
                project_base_dir,
            );
        }

        // User extensions from the agent dir
        add_resources(
            ResourceType::Extensions,
            collect_auto_extension_entries(&user_dirs.0),
            &user_metadata,
            &user_overrides.0,
            global_base_dir,
        );

        // User skills from the agent dir
        add_resources(
            ResourceType::Skills,
            collect_auto_skill_entries(&user_dirs.1, SkillDiscoveryMode::Pi),
            &user_metadata,
            &user_overrides.1,
            global_base_dir,
        );

        // User skills from ~/.agents/ (with its own baseDir)
        let user_agents_base_dir = user_agents_skills_dir
            .parent()
            .map(Path::to_string_lossy)
            .map(Cow::into_owned)
            .unwrap_or_default();
        let user_agents_metadata = PathMetadata {
            base_dir: Some(user_agents_base_dir.clone()),
            ..user_metadata.clone()
        };
        add_resources(
            ResourceType::Skills,
            collect_auto_skill_entries(&user_agents_skills_dir, SkillDiscoveryMode::Agents),
            &user_agents_metadata,
            &user_overrides.1,
            &user_agents_base_dir,
        );

        add_resources(
            ResourceType::Prompts,
            collect_auto_prompt_entries(&user_dirs.2),
            &user_metadata,
            &user_overrides.2,
            global_base_dir,
        );
        add_resources(
            ResourceType::Themes,
            collect_auto_theme_entries(&user_dirs.3),
            &user_metadata,
            &user_overrides.3,
            global_base_dir,
        );
    }

    /// The no-matching-package message, upstream's
    /// `buildNoMatchingPackageMessage`.
    fn build_no_matching_package_message(
        &self,
        source: &str,
        configured_packages: &[PackageSourceView],
    ) -> String {
        let suggestion = self.find_suggested_configured_source(source, configured_packages);
        suggestion.map_or_else(
            || format!("No matching package found for {source}"),
            |suggestion| {
                format!("No matching package found for {source}. Did you mean {suggestion}?")
            },
        )
    }

    /// The suggestion scan, upstream's `findSuggestedConfiguredSource`: an
    /// npm-less match set — crate names, git shorthands with and without
    /// refs.
    fn find_suggested_configured_source(
        &self,
        source: &str,
        configured_packages: &[PackageSourceView],
    ) -> Option<String> {
        let trimmed_source = source.trim();
        let mut suggestions: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

        for pkg in configured_packages {
            let source_str = pkg.source().to_string();
            let Ok(parsed) = self.parse_source(&source_str) else {
                continue;
            };
            match parsed {
                ParsedSource::Crate(crate_source) => {
                    if trimmed_source == crate_source.name {
                        suggestions.insert(source_str);
                    }
                }
                ParsedSource::Git(git_source) => {
                    let shorthand = format!("{}/{}", git_source.host, git_source.path);
                    let shorthand_with_ref = git_source
                        .r#ref
                        .as_ref()
                        .map(|reference| format!("{shorthand}@{reference}"));
                    if trimmed_source == shorthand
                        || shorthand_with_ref.is_some_and(|candidate| trimmed_source == candidate)
                    {
                        suggestions.insert(source_str);
                    }
                }
                _ => {}
            }
        }

        suggestions.into_iter().next()
    }
}

/// Shape the accumulator into the output, upstream's `toResolvedPaths`:
/// precedence-rank sort, then first-wins dedupe on the canonical path.
fn to_resolved_paths(accumulator: ResourceAccumulator) -> ResolvedPaths {
    let map_to_resolved =
        |entries: indexmap::IndexMap<PathBuf, AccumulatedResource>| -> Vec<ResolvedResource> {
            let mut resolved: Vec<ResolvedResource> = entries
                .into_iter()
                .map(|(path, entry)| ResolvedResource {
                    path: path.to_string_lossy().into_owned(),
                    enabled: entry.enabled,
                    metadata: entry.metadata,
                })
                .collect();
            resolved.sort_by_key(|entry| resource_precedence_rank(&entry.metadata));

            let mut seen = HashSet::new();
            resolved
                .into_iter()
                .filter(|entry| {
                    let canonical_path = crate::utils::paths::canonicalize_path(&entry.path);
                    seen.insert(canonical_path)
                })
                .collect()
        };

    ResolvedPaths {
        extensions: map_to_resolved(accumulator.extensions),
        skills: map_to_resolved(accumulator.skills),
        prompts: map_to_resolved(accumulator.prompts),
        themes: map_to_resolved(accumulator.themes),
    }
}

/// The git upstream shorthand, upstream's `@{upstream}` ref literal — the
/// braces are git's revision syntax, not a format placeholder.
const UPSTREAM_REF: &str = "@{upstream}";

// =============================================================================
// Git update machinery
// =============================================================================

impl<S: SettingsStorage + 'static> DefaultPackageManager<S> {
    /// Whether a git checkout's remote moved, upstream's
    /// `gitHasAvailableUpdate`: the local HEAD differs from the remote's.
    async fn git_has_available_update(&self, installed_path: &str) -> bool {
        if self.offline() {
            return false;
        }

        let run = async {
            let options = CommandRunOptions {
                cwd: Some(installed_path.to_string()),
                env: Vec::new(),
            };
            let local_head = self
                .command_runner
                .run_capture(
                    "git",
                    &["rev-parse".to_string(), "HEAD".to_string()],
                    &options,
                    Some(NETWORK_TIMEOUT_MS),
                )
                .await?;
            let remote_head = self.get_remote_git_head(installed_path).await?;
            Ok::<String, PackageManagerError>(format!("{local_head}|{remote_head}"))
        };
        run.await.is_ok_and(|pair| {
            let (local_head, remote_head) = pair.split_once('|').unwrap_or(("", ""));
            local_head.trim() != remote_head.trim()
        })
    }

    /// The remote HEAD the upstream ref answers, upstream's
    /// `getRemoteGitHead`: the upstream branch first, then the remote's
    /// HEAD symref.
    ///
    /// # Errors
    /// The git failure, or the missing remote HEAD.
    async fn get_remote_git_head(
        &self,
        installed_path: &str,
    ) -> Result<String, PackageManagerError> {
        let upstream_ref = self.get_git_upstream_ref(installed_path).await;
        if let Some(upstream_ref) = upstream_ref {
            let remote_head = self
                .run_git_remote_command(
                    installed_path,
                    &["ls-remote".to_string(), "origin".to_string(), upstream_ref],
                )
                .await?;
            if let Some(matched) = first_hex40_line(&remote_head) {
                return Ok(matched);
            }
        }

        let remote_head = self
            .run_git_remote_command(
                installed_path,
                &[
                    "ls-remote".to_string(),
                    "origin".to_string(),
                    "HEAD".to_string(),
                ],
            )
            .await?;
        let matched = remote_head
            .lines()
            .find_map(|line| {
                let (hash, refname) = line.split_once('\t')?;
                (refname == "HEAD" && hash.len() == 40).then(|| hash.to_string())
            })
            .ok_or_else(|| PackageManagerError("Failed to determine remote HEAD".to_string()))?;
        Ok(matched)
    }

    /// The local fetch/reset target for an unpinned checkout, upstream's
    /// `getLocalGitUpdateTarget`: the upstream branch when configured,
    /// else the remote's HEAD symref, else a hard HEAD fetch.
    ///
    /// # Errors
    /// The git failure when no target resolves.
    #[expect(
        clippy::too_many_lines,
        reason = "the fallback ladder (upstream branch, remote HEAD symref, hard HEAD fetch) reads as one sequence in upstream's getLocalGitUpdateTarget"
    )]
    async fn get_local_git_update_target(
        &self,
        installed_path: &str,
    ) -> Result<GitUpdateTarget, PackageManagerError> {
        let options = CommandRunOptions {
            cwd: Some(installed_path.to_string()),
            env: Vec::new(),
        };
        let upstream_probe = self
            .command_runner
            .run_capture(
                "git",
                &[
                    "rev-parse".to_string(),
                    "--abbrev-ref".to_string(),
                    UPSTREAM_REF.to_string(),
                ],
                &options,
                Some(NETWORK_TIMEOUT_MS),
            )
            .await;
        if let Ok(upstream) = upstream_probe {
            let trimmed_upstream = upstream.trim();
            if !trimmed_upstream.starts_with("origin/") {
                return Err(PackageManagerError(format!(
                    "Unsupported upstream remote: {trimmed_upstream}"
                )));
            }
            let branch = trimmed_upstream
                .strip_prefix("origin/")
                .unwrap_or_default()
                .to_string();
            if branch.is_empty() {
                return Err(PackageManagerError(
                    "Missing upstream branch name".to_string(),
                ));
            }
            let head = self
                .command_runner
                .run_capture(
                    "git",
                    &["rev-parse".to_string(), UPSTREAM_REF.to_string()],
                    &options,
                    Some(NETWORK_TIMEOUT_MS),
                )
                .await?;
            return Ok(GitUpdateTarget {
                reference: UPSTREAM_REF.to_string(),
                head,
                fetch_args: vec![
                    "fetch".to_string(),
                    "--prune".to_string(),
                    "--no-tags".to_string(),
                    "origin".to_string(),
                    format!("+refs/heads/{branch}:refs/remotes/origin/{branch}"),
                ],
            });
        }

        // The remote-head fallback: set the symref best-effort first,
        // upstream's swallowed `remote set-head` failure.
        let set_head = self
            .command_runner
            .run(
                "git",
                &[
                    "remote".to_string(),
                    "set-head".to_string(),
                    "origin".to_string(),
                    "-a".to_string(),
                ],
                &options,
            )
            .await;
        let _ = set_head;
        let head = self
            .command_runner
            .run_capture(
                "git",
                &["rev-parse".to_string(), "origin/HEAD".to_string()],
                &options,
                Some(NETWORK_TIMEOUT_MS),
            )
            .await?;
        let origin_head_ref = self
            .command_runner
            .run_capture(
                "git",
                &[
                    "symbolic-ref".to_string(),
                    "refs/remotes/origin/HEAD".to_string(),
                ],
                &options,
                Some(NETWORK_TIMEOUT_MS),
            )
            .await
            .unwrap_or_default();
        let branch = origin_head_ref
            .trim()
            .strip_prefix("refs/remotes/origin/")
            .unwrap_or_default()
            .to_string();
        if !branch.is_empty() {
            return Ok(GitUpdateTarget {
                reference: "origin/HEAD".to_string(),
                head,
                fetch_args: vec![
                    "fetch".to_string(),
                    "--prune".to_string(),
                    "--no-tags".to_string(),
                    "origin".to_string(),
                    format!("+refs/heads/{branch}:refs/remotes/origin/{branch}"),
                ],
            });
        }
        Ok(GitUpdateTarget {
            reference: "origin/HEAD".to_string(),
            head,
            fetch_args: vec![
                "fetch".to_string(),
                "--prune".to_string(),
                "--no-tags".to_string(),
                "origin".to_string(),
                "+HEAD:refs/remotes/origin/HEAD".to_string(),
            ],
        })
    }

    /// The checkout's upstream branch ref, upstream's `getGitUpstreamRef`.
    async fn get_git_upstream_ref(&self, installed_path: &str) -> Option<String> {
        let options = CommandRunOptions {
            cwd: Some(installed_path.to_string()),
            env: Vec::new(),
        };
        let upstream = self
            .command_runner
            .run_capture(
                "git",
                &[
                    "rev-parse".to_string(),
                    "--abbrev-ref".to_string(),
                    UPSTREAM_REF.to_string(),
                ],
                &options,
                Some(NETWORK_TIMEOUT_MS),
            )
            .await
            .ok()?;
        let trimmed = upstream.trim();
        let branch = trimmed.strip_prefix("origin/")?;
        (!branch.is_empty()).then(|| format!("refs/heads/{branch}"))
    }

    /// A remote-scoped git command, upstream's `runGitRemoteCommand`: the
    /// `GIT_TERMINAL_PROMPT` off switch keeps a missing credential from
    /// hanging the check.
    async fn run_git_remote_command(
        &self,
        installed_path: &str,
        args: &[String],
    ) -> Result<String, PackageManagerError> {
        let options = CommandRunOptions {
            cwd: Some(installed_path.to_string()),
            env: vec![("GIT_TERMINAL_PROMPT".to_string(), "0".to_string())],
        };
        self.command_runner
            .run_capture("git", args, &options, Some(NETWORK_TIMEOUT_MS))
            .await
    }

    /// The installed receipt version against the configured pin, upstream's
    /// `installedNpmMatchesConfiguredVersion` with the range machinery
    /// dropped: a pinned spec matches its exact version, an unpinned one
    /// matches anything installed.
    fn installed_crate_matches_configured_version(
        source: &CrateSource,
        installed_path: &str,
    ) -> bool {
        let Some(installed_version) = Self::get_installed_crate_version(installed_path) else {
            return false;
        };
        source
            .version
            .as_ref()
            .is_none_or(|version| &installed_version == version)
    }
}

/// The git fetch/reset target, upstream's `{ ref, head, fetchArgs }`.
struct GitUpdateTarget {
    /// The ref the reset resolves, upstream's `ref`.
    reference: String,
    /// The captured head, upstream's `head`.
    #[allow(
        dead_code,
        reason = "the upstream shape carries it; the reset re-resolves through git"
    )]
    head: String,
    /// The fetch arguments, upstream's `fetchArgs`.
    fetch_args: Vec<String>,
}

/// The first 40-hex line prefix of a ls-remote output, upstream's
/// `/^([0-9a-f]{40})\s+/m` match.
fn first_hex40_line(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let hash = line.get(0..40)?;
        (hash.chars().all(|c| c.is_ascii_hexdigit()) && line[40..].starts_with([' ', '\t']))
            .then(|| hash.to_string())
    })
}

impl<S: SettingsStorage> std::fmt::Debug for PackageManagerOptions<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackageManagerOptions")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .finish_non_exhaustive()
    }
}

impl<S: SettingsStorage> std::fmt::Debug for DefaultPackageManager<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultPackageManager")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .finish_non_exhaustive()
    }
}

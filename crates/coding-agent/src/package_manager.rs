//! The package manager's static resource resolution, upstream's
//! `src/core/package-manager.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! This slice carries the resolution the resource loader consumes: the
//! settings-driven local entries, the auto-discovered global
//! (`~/.pi/agent/{skills,prompts,themes,extensions}` plus
//! `~/.agents/skills`) and project (`.pi/{...}`) directories, the
//! `!`/`+`/`-` override and glob-pattern machinery over them, the
//! precedence rank that orders name-collision resolution, and the local
//! arm of package sources (a path entry pointing at a file or a package
//! directory with a `pi` manifest). The npm/git install arm — `resolve`'s
//! `resolvePackageSources` for npm and git sources, the install/remove/
//! update commands, and the progress surface — rides the package-manager
//! ticket (#129): until it lands, a settings package the install machinery
//! would resolve is skipped, the port's documented degradation of
//! upstream's install-and-proceed behavior.
//!
//! The ignore machinery (npm `ignore` semantics) is the workspace's single
//! restatement, `pi_agent_core::harness::skills::{IgnoreMatcher,
//! prefix_ignore_pattern}`; upstream re-implements it here per file.

use std::collections::HashSet;
use std::path::Path;

use indexmap::IndexMap;

use pi_agent_core::harness::skills::{IgnoreMatcher, prefix_ignore_pattern};

use crate::config::{CONFIG_DIR_NAME, home_dir};
use crate::pi_manifest::{PiManifest, read_pi_manifest};
use crate::settings_manager::{Settings, SettingsManager, SettingsStorage};
use crate::source_info::{SourceOrigin, SourceScope};
use crate::utils::minimatch::matches as minimatch_matches;
use crate::utils::paths::{
    PathInputOptions, basename_posix, canonicalize_path, dirname_posix, relative_posix,
    resolve_path_with,
};

// =============================================================================
// Types, upstream's PathMetadata / ResolvedResource / ResolvedPaths
// =============================================================================

/// A resource's package metadata, upstream's `PathMetadata`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathMetadata {
    /// The source label (`local`, `auto`, `npm:<name>`, `git:<url>`,
    /// `extension:<name>`, the raw source string for package entries).
    pub source: String,
    /// The trust scope the resource loads under.
    pub scope: SourceScope,
    /// Whether a package manifest named the resource.
    pub origin: SourceOrigin,
    /// The directory the resource was declared against.
    pub base_dir: Option<String>,
}

/// One resolved resource path with its metadata and enabled state,
/// upstream's `ResolvedResource`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedResource {
    /// The resolved absolute path.
    pub path: String,
    /// Whether the resource survives the settings' override patterns.
    pub enabled: bool,
    /// The resource's package metadata.
    pub metadata: PathMetadata,
}

/// The resolved resources per type, upstream's `ResolvedPaths`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedPaths {
    /// Extension entry files.
    pub extensions: Vec<ResolvedResource>,
    /// Skill files (SKILL.md paths or root `.md` files).
    pub skills: Vec<ResolvedResource>,
    /// Prompt template files.
    pub prompts: Vec<ResolvedResource>,
    /// Theme JSON files.
    pub themes: Vec<ResolvedResource>,
}

/// The resource families, upstream's `ResourceType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// Extensions.
    Extensions,
    /// Skills.
    Skills,
    /// Prompt templates.
    Prompts,
    /// Themes.
    Themes,
}

/// The resource families in upstream's `RESOURCE_TYPES` order.
pub const RESOURCE_TYPES: [ResourceType; 4] = [
    ResourceType::Extensions,
    ResourceType::Skills,
    ResourceType::Prompts,
    ResourceType::Themes,
];

impl ResourceType {
    /// The settings key and directory name the family shares, upstream's
    /// `resourceType` string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Extensions => "extensions",
            Self::Skills => "skills",
            Self::Prompts => "prompts",
            Self::Themes => "themes",
        }
    }
}

/// The accumulator entry, upstream's
/// `Map<string, { metadata, enabled }>`. An `IndexMap` restates the JS
/// `Map`'s insertion order with first-write-wins.
type TargetMap = IndexMap<String, (PathMetadata, bool)>;

/// The per-type resource accumulator, upstream's `ResourceAccumulator`.
#[derive(Debug, Default)]
struct ResourceAccumulator {
    extensions: TargetMap,
    skills: TargetMap,
    prompts: TargetMap,
    themes: TargetMap,
}

impl ResourceAccumulator {
    const fn target(&mut self, resource_type: ResourceType) -> &mut TargetMap {
        match resource_type {
            ResourceType::Extensions => &mut self.extensions,
            ResourceType::Skills => &mut self.skills,
            ResourceType::Prompts => &mut self.prompts,
            ResourceType::Themes => &mut self.themes,
        }
    }
}

// =============================================================================
// Path classification, upstream's parseSource's local arm
// =============================================================================

/// Whether a settings package source is a local path, upstream's
/// `parseSource` classification.
///
/// An `npm:` spec is npm, a path (`isLocalPath`) is local, a parseable
/// git URL is git, and anything else falls back to local — the port's
/// check mirrors that order.
#[must_use]
pub fn is_local_source(source: &str) -> bool {
    if source.starts_with("npm:") {
        return false;
    }
    if crate::utils::paths::is_local_path(source) {
        return true;
    }
    crate::utils::git::parse_git_url(source).is_none()
}

// =============================================================================
// Pattern helpers, upstream's isPattern / splitPatterns / applyPatterns
// =============================================================================

/// Whether an entry is a pattern rather than a plain path, upstream's
/// `isPattern`.
#[must_use]
fn is_pattern(entry: &str) -> bool {
    entry.starts_with('!')
        || entry.starts_with('+')
        || entry.starts_with('-')
        || entry.contains('*')
        || entry.contains('?')
}

/// Whether an entry is an override (`!`/`+`/`-`), upstream's
/// `isOverridePattern`.
#[must_use]
fn is_override_pattern(entry: &str) -> bool {
    entry.starts_with('!') || entry.starts_with('+') || entry.starts_with('-')
}

/// Whether an entry carries a glob metacharacter, upstream's
/// `hasGlobPattern`.
#[must_use]
fn has_glob_pattern(entry: &str) -> bool {
    entry.contains('*') || entry.contains('?')
}

/// The plain entries and the patterns of one settings array, upstream's
/// `splitPatterns`.
#[must_use]
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

/// Minimatch one candidate against one pattern, upstream's `minimatch`
/// with its case-sensitive default.
fn minimatch(candidate: &str, pattern: &str) -> bool {
    minimatch_matches(candidate, pattern, false)
}

/// Whether a path matches any glob pattern against its relative form, its
/// basename, its absolute POSIX form — and, for SKILL.md files, its
/// parent directory's relative form, basename, and absolute POSIX form —
/// upstream's `matchesAnyPattern`.
#[must_use]
fn matches_any_pattern(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    let rel = relative_posix(base_dir, file_path);
    let name = basename_posix(file_path);
    let file_path_posix = to_posix_path(file_path);
    let is_skill_file = name == "SKILL.md";
    let (parent_rel, parent_name, parent_dir_posix) = if is_skill_file {
        let parent_dir = dirname_posix(file_path);
        (
            Some(relative_posix(base_dir, &parent_dir)),
            Some(basename_posix(&parent_dir)),
            Some(to_posix_path(&parent_dir)),
        )
    } else {
        (None, None, None)
    };

    patterns.iter().any(|pattern| {
        let normalized_pattern = to_posix_path(pattern);
        if minimatch(&rel, &normalized_pattern)
            || minimatch(&name, &normalized_pattern)
            || minimatch(&file_path_posix, &normalized_pattern)
        {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        minimatch(
            parent_rel.as_deref().unwrap_or_default(),
            &normalized_pattern,
        ) || minimatch(
            parent_name.as_deref().unwrap_or_default(),
            &normalized_pattern,
        ) || minimatch(
            parent_dir_posix.as_deref().unwrap_or_default(),
            &normalized_pattern,
        )
    })
}

/// Normalize an exact entry: a leading `./` drops, separators go POSIX,
/// upstream's `normalizeExactPattern`.
#[must_use]
fn normalize_exact_pattern(pattern: &str) -> String {
    let normalized = if pattern.starts_with("./") || pattern.starts_with(".\\") {
        &pattern[2..]
    } else {
        pattern
    };
    to_posix_path(normalized)
}

/// Whether a path matches any exact entry — against its relative form or
/// its absolute POSIX form, and for SKILL.md files its parent directory's
/// too — upstream's `matchesAnyExactPattern`.
#[must_use]
fn matches_any_exact_pattern(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    if patterns.is_empty() {
        return false;
    }
    let rel = relative_posix(base_dir, file_path);
    let name = basename_posix(file_path);
    let file_path_posix = to_posix_path(file_path);
    let is_skill_file = name == "SKILL.md";
    let (parent_rel, parent_dir_posix) = if is_skill_file {
        let parent_dir = dirname_posix(file_path);
        (
            Some(relative_posix(base_dir, &parent_dir)),
            Some(to_posix_path(&parent_dir)),
        )
    } else {
        (None, None)
    };

    patterns.iter().any(|pattern| {
        let normalized = normalize_exact_pattern(pattern);
        if normalized == rel || normalized == file_path_posix {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        normalized == parent_rel.as_deref().unwrap_or_default()
            || normalized == parent_dir_posix.as_deref().unwrap_or_default()
    })
}

/// The override entries of a settings array, upstream's
/// `getOverridePatterns`.
#[must_use]
fn get_override_patterns(entries: &[String]) -> Vec<String> {
    entries
        .iter()
        .filter(|pattern| is_override_pattern(pattern))
        .cloned()
        .collect()
}

/// Whether a resource path survives the settings' override patterns,
/// upstream's `isEnabledByOverrides`: `!` glob excludes disable, `+`
/// exact force-includes re-enable, `-` exact force-excludes disable
/// last.
#[must_use]
fn is_enabled_by_overrides(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    let overrides = get_override_patterns(patterns);
    let excludes: Vec<String> = overrides
        .iter()
        .filter(|p| p.starts_with('!'))
        .map(|p| p[1..].to_string())
        .collect();
    let force_includes: Vec<String> = overrides
        .iter()
        .filter(|p| p.starts_with('+'))
        .map(|p| p[1..].to_string())
        .collect();
    let force_excludes: Vec<String> = overrides
        .iter()
        .filter(|p| p.starts_with('-'))
        .map(|p| p[1..].to_string())
        .collect();

    let has_excludes = !excludes.is_empty();
    if !force_excludes.is_empty() && matches_any_exact_pattern(file_path, &force_excludes, base_dir)
    {
        // A force-exclude wins last, upstream's step order.
        return false;
    }
    if !force_includes.is_empty() && matches_any_exact_pattern(file_path, &force_includes, base_dir)
    {
        return true;
    }
    !(has_excludes && matches_any_pattern(file_path, &excludes, base_dir))
}

/// Apply an entry list to the collected paths, upstream's
/// `applyPatterns`. Plain patterns include; `!` patterns exclude; `+`
/// exact entries force-include from the full set; `-` exact entries
/// force-exclude. With no plain patterns everything starts enabled.
#[must_use]
fn apply_patterns(all_paths: &[String], patterns: &[String], base_dir: &str) -> Vec<String> {
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

    let mut result: Vec<String> = if includes.is_empty() {
        all_paths.to_vec()
    } else {
        all_paths
            .iter()
            .filter(|p| matches_any_pattern(p, &includes, base_dir))
            .cloned()
            .collect()
    };

    if !excludes.is_empty() {
        result.retain(|p| !matches_any_pattern(p, &excludes, base_dir));
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
        result.retain(|p| !matches_any_exact_pattern(p, &force_excludes, base_dir));
    }

    result
}

// =============================================================================
// Ignore rules, upstream's prefixIgnorePattern / addIgnoreRules
// =============================================================================

/// Add one directory's ignore-file patterns to the matcher, upstream's
/// `addIgnoreRules`.
///
/// Patterns prefix with the directory's path relative to the traversal
/// root so a subdirectory's ignore file applies at its own depth.
fn add_ignore_rules(matcher: &mut IgnoreMatcher, dir: &str, root_dir: &str) {
    let relative_dir = relative_posix(root_dir, dir);
    let prefix = if relative_dir.is_empty() {
        String::new()
    } else {
        format!("{relative_dir}/")
    };

    for filename in IGNORE_FILE_NAMES {
        let ignore_path = Path::new(dir).join(filename);
        let Ok(content) = std::fs::read_to_string(&ignore_path) else {
            continue;
        };
        let patterns: Vec<String> = content
            .lines()
            .filter_map(|line| prefix_ignore_pattern(line, &prefix))
            .collect();
        if !patterns.is_empty() {
            matcher.add(&patterns);
        }
    }
}

/// The ignore files each traversed directory contributes, upstream's
/// `IGNORE_FILE_NAMES`.
const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

/// Convert to POSIX separators, upstream's `toPosixPath`.
#[must_use]
fn to_posix_path(p: &str) -> String {
    p.replace('\\', "/")
}

// =============================================================================
// Directory collectors, upstream's collectFiles / collectSkillEntries /
// collectAutoPromptEntries / collectAutoThemeEntries /
// resolveExtensionEntries / collectAutoExtensionEntries
// =============================================================================

/// Whether a file name matches the resource family's extension filter,
/// upstream's `FILE_PATTERNS` regexes (`/\.(ts|js)$/`, `/\.md$/`,
/// `/\.json$/`).
///
/// The comparisons stay case-sensitive, upstream's regexes.
#[must_use]
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's FILE_PATTERNS regexes are case-sensitive; the 1:1 shape keeps it"
)]
fn matches_file_pattern(resource_type: ResourceType, name: &str) -> bool {
    match resource_type {
        ResourceType::Extensions => name.ends_with(".ts") || name.ends_with(".js"),
        ResourceType::Skills | ResourceType::Prompts => name.ends_with(".md"),
        ResourceType::Themes => name.ends_with(".json"),
    }
}

/// A skill directory's discovery flavor, upstream's `SkillDiscoveryMode`:
/// `pi` loads a directory's root `.md` files only at the traversal root,
/// `agents` only below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkillDiscoveryMode {
    /// The `.pi` flavor: root `.md` files load at the traversal root.
    Pi,
    /// The `.agents` flavor: root `.md` files load below the root.
    Agents,
}

/// Recursively collect a directory's files matching the family's
/// extension filter, upstream's `collectFiles`. Dot directories and
/// `node_modules` skip; ignore files filter the walk through one matcher
/// per traversal root.
fn collect_files(
    dir: &str,
    resource_type: ResourceType,
    matcher: &mut IgnoreMatcher,
    root_dir: &str,
) -> Vec<String> {
    let mut files = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return files;
    };

    add_ignore_rules(matcher, dir, root_dir);

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let (is_dir, is_file) = if file_type.is_symlink() {
            match crate::utils::paths::symlink_resolved_kind(&entry.path()) {
                Some(kind) => kind,
                None => continue,
            }
        } else {
            (file_type.is_dir(), file_type.is_file())
        };

        let full_path = entry.path().to_string_lossy().into_owned();
        let rel_path = relative_posix(root_dir, &full_path);
        let ignore_path = if is_dir {
            format!("{rel_path}/")
        } else {
            rel_path
        };
        if matcher.ignores(&ignore_path) {
            continue;
        }

        if is_dir {
            files.extend(collect_files(&full_path, resource_type, matcher, root_dir));
        } else if is_file && matches_file_pattern(resource_type, &name) {
            files.push(full_path);
        }
    }

    files
}

/// Collect a directory's skill entries, upstream's `collectSkillEntries`.
///
/// A `SKILL.md` at a directory's listing front owns the directory: it
/// loads and the walk stops for that directory. Otherwise subdirectories
/// recurse and root `.md` files load per the flavor's depth rule.
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's md ends-with probe is case-sensitive; the 1:1 shape keeps it"
)]
fn collect_skill_entries(
    dir: &str,
    mode: SkillDiscoveryMode,
    matcher: &mut IgnoreMatcher,
    root_dir: &str,
) -> Vec<String> {
    let mut entries_out = Vec::new();
    let Ok(read) = std::fs::read_dir(dir) else {
        return entries_out;
    };
    let dir_entries: Vec<std::fs::DirEntry> = read.flatten().collect();

    add_ignore_rules(matcher, dir, root_dir);

    // The first listed SKILL.md owns the directory, upstream's in-loop
    // `return`.
    for entry in &dir_entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != "SKILL.md" {
            continue;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let is_file = if file_type.is_symlink() {
            match crate::utils::paths::symlink_resolved_kind(&entry.path()) {
                Some((_, is_file)) => is_file,
                None => continue,
            }
        } else {
            file_type.is_file()
        };
        let full_path = entry.path().to_string_lossy().into_owned();
        let rel_path = relative_posix(root_dir, &full_path);
        if is_file && !matcher.ignores(&rel_path) {
            entries_out.push(full_path);
            return entries_out;
        }
    }

    for entry in &dir_entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let (is_dir, is_file) = if file_type.is_symlink() {
            match crate::utils::paths::symlink_resolved_kind(&entry.path()) {
                Some(kind) => kind,
                None => continue,
            }
        } else {
            (file_type.is_dir(), file_type.is_file())
        };

        let full_path = entry.path().to_string_lossy().into_owned();
        let rel_path = relative_posix(root_dir, &full_path);
        let should_include_markdown_file = is_file
            && name.ends_with(".md")
            && !matcher.ignores(&rel_path)
            && ((mode == SkillDiscoveryMode::Pi && dir == root_dir)
                || (mode == SkillDiscoveryMode::Agents && dir != root_dir));
        if should_include_markdown_file {
            entries_out.push(full_path);
            continue;
        }

        if !is_dir {
            continue;
        }
        if matcher.ignores(&format!("{rel_path}/")) {
            continue;
        }

        entries_out.extend(collect_skill_entries(&full_path, mode, matcher, root_dir));
    }

    entries_out
}

/// The walk-up git-repo root, upstream's `findGitRepoRoot`: the nearest
/// ancestor (or start) holding a `.git` entry.
#[must_use]
pub fn find_git_repo_root(start_dir: &str) -> Option<String> {
    let mut dir = resolve_default(start_dir);
    loop {
        if Path::new(dir.as_str()).join(".git").exists() {
            return Some(dir);
        }
        dir = parent_dir(&dir)?;
    }
}

/// The skill entries behind one auto-discovered skills directory,
/// upstream's `collectAutoSkillEntries`.
#[must_use]
pub fn collect_auto_skill_entries(dir: &str, mode: SkillDiscoveryMode) -> Vec<String> {
    let mut matcher = IgnoreMatcher::new();
    collect_skill_entries(dir, mode, &mut matcher, dir)
}

/// The `.agents/skills` directories from the start dir up to its git repo
/// root, upstream's `collectAncestorAgentsSkillDirs`.
#[must_use]
pub fn collect_ancestor_agents_skill_dirs(start_dir: &str) -> Vec<String> {
    let mut skill_dirs = Vec::new();
    let resolved_start_dir = resolve_default(start_dir);
    let git_repo_root = find_git_repo_root(&resolved_start_dir);

    let mut dir = resolved_start_dir;
    loop {
        skill_dirs.push(
            Path::new(&dir)
                .join(".agents")
                .join("skills")
                .to_string_lossy()
                .into_owned(),
        );
        if git_repo_root.as_deref() == Some(dir.as_str()) {
            break;
        }
        let Some(parent) = parent_dir(&dir) else {
            break;
        };
        dir = parent;
    }

    skill_dirs
}

/// Collect a directory's direct `.md` children, upstream's
/// `collectAutoPromptEntries`.
#[must_use]
pub fn collect_auto_prompt_entries(dir: &str) -> Vec<String> {
    collect_flat_entries(dir, ResourceType::Prompts)
}

/// Collect a directory's direct `.json` children, upstream's
/// `collectAutoThemeEntries`.
#[must_use]
pub fn collect_auto_theme_entries(dir: &str) -> Vec<String> {
    collect_flat_entries(dir, ResourceType::Themes)
}

/// The flat collector both auto-entry readers share: direct children only,
/// dot files and `node_modules` skip, ignore files filter.
fn collect_flat_entries(dir: &str, resource_type: ResourceType) -> Vec<String> {
    let mut entries = Vec::new();
    let Ok(read) = std::fs::read_dir(dir) else {
        return entries;
    };

    let mut matcher = IgnoreMatcher::new();
    add_ignore_rules(&mut matcher, dir, dir);

    for entry in read.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let is_file = if file_type.is_symlink() {
            match crate::utils::paths::symlink_resolved_kind(&entry.path()) {
                Some((_, is_file)) => is_file,
                None => continue,
            }
        } else {
            file_type.is_file()
        };

        let full_path = entry.path().to_string_lossy().into_owned();
        let rel_path = relative_posix(dir, &full_path);
        if matcher.ignores(&rel_path) {
            continue;
        }

        if is_file && matches_file_pattern(resource_type, &name) {
            entries.push(full_path);
        }
    }

    entries
}

/// A directory's extension entry files, upstream's
/// `resolveExtensionEntries`.
///
/// The `pi.extensions` manifest entries when the directory's
/// `package.json` declares any that exist, else `index.ts`, else
/// `index.js`.
#[must_use]
fn resolve_extension_entries(dir: &str) -> Option<Vec<String>> {
    let package_json_path = Path::new(dir).join("package.json");
    if package_json_path.exists() {
        let manifest = read_pi_manifest(&package_json_path.to_string_lossy());
        if let Some(manifest) = manifest
            && let Some(extensions) = manifest.extensions
            && !extensions.is_empty()
        {
            let entries: Vec<String> = extensions
                .iter()
                .map(|ext_path| Path::new(dir).join(ext_path).to_string_lossy().into_owned())
                .filter(|p| Path::new(p.as_str()).exists())
                .collect();
            if !entries.is_empty() {
                return Some(entries);
            }
        }
    }

    let ts_index = Path::new(dir).join("index.ts");
    if ts_index.exists() {
        return Some(vec![ts_index.to_string_lossy().into_owned()]);
    }
    let js_index = Path::new(dir).join("index.js");
    if js_index.exists() {
        return Some(vec![js_index.to_string_lossy().into_owned()]);
    }

    None
}

/// Collect a directory's extension entry files, upstream's
/// `collectAutoExtensionEntries`.
///
/// The directory's own explicit entries when it declares any, else one
/// entry per direct `.ts`/`.js` child and per child directory's own
/// explicit entries.
#[must_use]
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream ends-with probes for the ts and js extensions are case-sensitive; the 1:1 shape keeps it"
)]
pub fn collect_auto_extension_entries(dir: &str) -> Vec<String> {
    if let Some(root_entries) = resolve_extension_entries(dir) {
        return root_entries;
    }

    let mut entries = Vec::new();
    let Ok(read) = std::fs::read_dir(dir) else {
        return entries;
    };

    let mut matcher = IgnoreMatcher::new();
    add_ignore_rules(&mut matcher, dir, dir);

    for entry in read.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let (is_dir, is_file) = if file_type.is_symlink() {
            match crate::utils::paths::symlink_resolved_kind(&entry.path()) {
                Some(kind) => kind,
                None => continue,
            }
        } else {
            (file_type.is_dir(), file_type.is_file())
        };

        let full_path = entry.path().to_string_lossy().into_owned();
        let rel_path = relative_posix(dir, &full_path);
        let ignore_path = if is_dir {
            format!("{rel_path}/")
        } else {
            rel_path
        };
        if matcher.ignores(&ignore_path) {
            continue;
        }

        let is_extension_file = name.ends_with(".ts") || name.ends_with(".js");
        if is_file && is_extension_file {
            entries.push(full_path);
        } else if is_dir && let Some(resolved_entries) = resolve_extension_entries(&full_path) {
            entries.extend(resolved_entries);
        }
    }

    entries
}

/// Collect a package directory's resource files per family, upstream's
/// `collectResourceFiles`.
///
/// Skills ride the skill-entry collector, extensions the
/// extension-entry collector, the rest the recursive file collector.
#[must_use]
pub fn collect_resource_files(dir: &str, resource_type: ResourceType) -> Vec<String> {
    let mut matcher = IgnoreMatcher::new();
    match resource_type {
        ResourceType::Skills => {
            collect_skill_entries(dir, SkillDiscoveryMode::Pi, &mut matcher, dir)
        }
        ResourceType::Extensions => collect_auto_extension_entries(dir),
        ResourceType::Prompts | ResourceType::Themes => {
            collect_files(dir, resource_type, &mut matcher, dir)
        }
    }
}

/// Expand a glob entry against a root and drop dot segments, upstream's
/// `expandPackageGlob` over npm `globSync`. Results sort ascending.
#[must_use]
fn expand_package_glob(pattern: &str, root: &str) -> Vec<String> {
    let full = format!("{root}/{pattern}");
    let Ok(paths) = glob::glob(&full) else {
        return Vec::new();
    };
    let mut matches: Vec<String> = paths
        .flatten()
        .map(|path| path.to_string_lossy().into_owned())
        .filter(|path| {
            relative_posix(root, path)
                .split('/')
                .all(|segment| segment == ".." || !segment.starts_with('.'))
        })
        .collect();
    matches.sort();
    matches
}

// =============================================================================
// Precedence, upstream's resourcePrecedenceRank
// =============================================================================

/// The numeric precedence rank of a resource's metadata, upstream's
/// `resourcePrecedenceRank`.
///
/// Lower wins name collisions: project settings entries (0), project
/// auto-discovered (1), user settings entries (2), user auto-discovered
/// (3), package resources (4).
#[must_use]
pub fn resource_precedence_rank(metadata: &PathMetadata) -> u8 {
    if metadata.origin == SourceOrigin::Package {
        return 4;
    }
    let scope_base: u8 = if metadata.scope == SourceScope::Project {
        0
    } else {
        2
    };
    scope_base + u8::from(metadata.source != "local")
}

// =============================================================================
// DefaultPackageManager, upstream's class
// =============================================================================

/// The static resource resolver, upstream's `DefaultPackageManager` with
/// its install arm deferred to #129.
///
/// Settings-driven local entries, the auto-discovered directories, the
/// local arm of package sources, and the resolved-path assembly. The
/// settings manager rides each call — upstream's constructor-held
/// instance — because the loader owns it and shares it by borrow.
#[derive(Clone, Debug)]
pub struct DefaultPackageManager {
    cwd: String,
    agent_dir: String,
}

/// The constructor options, upstream's `PackageManagerOptions` minus the
/// settings manager the calls now borrow.
#[derive(Debug)]
pub struct DefaultPackageManagerOptions {
    /// The working directory.
    pub cwd: String,
    /// The agent config directory.
    pub agent_dir: String,
}

impl DefaultPackageManager {
    /// The manager, upstream's `new DefaultPackageManager(options)`.
    #[must_use]
    pub fn new(options: &DefaultPackageManagerOptions) -> Self {
        Self {
            cwd: resolve_default(&options.cwd),
            agent_dir: resolve_default(&options.agent_dir),
        }
    }

    /// Resolve every resource family, upstream's `resolve` with the
    /// npm/git install arm deferred (#129).
    ///
    /// Project settings entries apply before global ones, then the
    /// auto-discovered directories (project gated on the project being
    /// trusted, then user), then the assembly sorts by precedence rank and
    /// drops canonical duplicates.
    #[must_use]
    pub fn resolve<S: SettingsStorage>(
        &self,
        settings_manager: &SettingsManager<S>,
    ) -> ResolvedPaths {
        let mut accumulator = ResourceAccumulator::default();
        let global_settings = settings_manager.get_global_settings();
        let project_settings = settings_manager.get_project_settings();

        // Package sources, upstream's resolvePackageSources. Local sources
        // resolve statically; npm/git sources ride #129.
        let all_packages: Vec<(String, SourceScope)> =
            package_sources(&project_settings, SourceScope::Project)
                .into_iter()
                .chain(package_sources(&global_settings, SourceScope::User))
                .collect();
        for (source, scope) in all_packages {
            if !is_local_source(&source) {
                continue;
            }
            let metadata = PathMetadata {
                source: source.clone(),
                scope,
                origin: SourceOrigin::Package,
                base_dir: None,
            };
            self.resolve_local_extension_source(
                &source,
                &mut accumulator,
                &metadata,
                &self.base_dir_for_scope(scope),
            );
        }

        let global_base_dir = self.agent_dir.clone();
        let project_base_dir = Path::new(&self.cwd)
            .join(CONFIG_DIR_NAME)
            .to_string_lossy()
            .into_owned();

        for resource_type in RESOURCE_TYPES {
            let project_entries = settings_strings(&project_settings, resource_type.as_str());
            let global_entries = settings_strings(&global_settings, resource_type.as_str());
            self.resolve_local_entries(
                &project_entries,
                resource_type,
                &mut accumulator,
                &PathMetadata {
                    source: "local".to_string(),
                    scope: SourceScope::Project,
                    origin: SourceOrigin::TopLevel,
                    base_dir: None,
                },
                &project_base_dir,
            );
            self.resolve_local_entries(
                &global_entries,
                resource_type,
                &mut accumulator,
                &PathMetadata {
                    source: "local".to_string(),
                    scope: SourceScope::User,
                    origin: SourceOrigin::TopLevel,
                    base_dir: None,
                },
                &global_base_dir,
            );
        }

        self.add_auto_discovered_resources(
            &mut accumulator,
            &global_settings,
            &project_settings,
            settings_manager.is_project_trusted(),
            &global_base_dir,
            &project_base_dir,
        );

        to_resolved_paths(accumulator)
    }

    /// Resolve CLI extension sources, upstream's `resolveExtensionSources`
    /// with the `temporary` scope the loader always asks for and the
    /// npm/git arm deferred (#129).
    #[must_use]
    pub fn resolve_extension_sources(&self, sources: &[String]) -> ResolvedPaths {
        let mut accumulator = ResourceAccumulator::default();
        for source in sources {
            if !is_local_source(source) {
                continue;
            }
            let metadata = PathMetadata {
                source: source.clone(),
                scope: SourceScope::Temporary,
                origin: SourceOrigin::Package,
                base_dir: None,
            };
            self.resolve_local_extension_source(
                source,
                &mut accumulator,
                &metadata,
                &self.base_dir_for_scope(SourceScope::Temporary),
            );
        }
        to_resolved_paths(accumulator)
    }

    /// The base directory a scope's entries resolve against, upstream's
    /// `getBaseDirForScope`.
    #[must_use]
    fn base_dir_for_scope(&self, scope: SourceScope) -> String {
        match scope {
            SourceScope::User => self.agent_dir.clone(),
            SourceScope::Project | SourceScope::Temporary => self.cwd.clone(),
        }
    }

    /// Resolve one local package source, upstream's
    /// `resolveLocalExtensionSource`: a file becomes one extension entry,
    /// a directory collects its package resources (or becomes one entry
    /// when it ships none).
    fn resolve_local_extension_source(
        &self,
        source: &str,
        accumulator: &mut ResourceAccumulator,
        metadata: &PathMetadata,
        base_dir: &str,
    ) {
        let resolved = self.resolve_path_from_base(source, base_dir);
        if !resolved.exists() {
            return;
        }

        let Ok(fs_meta) = std::fs::metadata(resolved.as_path()) else {
            return;
        };
        if fs_meta.is_file() {
            let mut metadata = metadata.clone();
            metadata.base_dir = Some(dirname_posix(&resolved.to_string_lossy()));
            Self::add_resource(
                &mut accumulator.extensions,
                &resolved.to_string_lossy(),
                &metadata,
                true,
            );
            return;
        }
        if fs_meta.is_dir() {
            let mut metadata = metadata.clone();
            metadata.base_dir = Some(resolved.to_string_lossy().into_owned());
            let resources = Self::collect_package_resources(
                &resolved.to_string_lossy(),
                accumulator,
                &metadata,
            );
            if !resources {
                Self::add_resource(
                    &mut accumulator.extensions,
                    &resolved.to_string_lossy(),
                    &metadata,
                    true,
                );
            }
        }
    }

    /// Collect one package root's resources, upstream's
    /// `collectPackageResources` with its filter arm deferred (#129): a
    /// `pi` manifest's entries when one exists, else the default layout
    /// directories.
    fn collect_package_resources(
        package_root: &str,
        accumulator: &mut ResourceAccumulator,
        metadata: &PathMetadata,
    ) -> bool {
        let package_json_path = Path::new(package_root).join("package.json");
        let manifest: Option<PiManifest> = if package_json_path.exists() {
            read_pi_manifest(&package_json_path.to_string_lossy())
        } else {
            None
        };
        if let Some(manifest) = manifest {
            Self::add_manifest_entries(
                manifest.extensions.as_deref(),
                package_root,
                ResourceType::Extensions,
                accumulator,
                metadata,
            );
            Self::add_manifest_entries(
                manifest.skills.as_deref(),
                package_root,
                ResourceType::Skills,
                accumulator,
                metadata,
            );
            Self::add_manifest_entries(
                manifest.prompts.as_deref(),
                package_root,
                ResourceType::Prompts,
                accumulator,
                metadata,
            );
            Self::add_manifest_entries(
                manifest.themes.as_deref(),
                package_root,
                ResourceType::Themes,
                accumulator,
                metadata,
            );
            return true;
        }

        let mut has_any_dir = false;
        for resource_type in RESOURCE_TYPES {
            let dir = Path::new(package_root).join(resource_type.as_str());
            if dir.exists() {
                let files = collect_resource_files(&dir.to_string_lossy(), resource_type);
                for f in files {
                    Self::add_resource(accumulator.target(resource_type), &f, metadata, true);
                }
                has_any_dir = true;
            }
        }
        has_any_dir
    }

    /// Add a manifest's entries for one family, upstream's
    /// `addManifestEntries`: plain entries resolve against the root and
    /// collect, override patterns filter what stays enabled.
    fn add_manifest_entries(
        entries: Option<&[String]>,
        root: &str,
        resource_type: ResourceType,
        accumulator: &mut ResourceAccumulator,
        metadata: &PathMetadata,
    ) {
        let Some(entries) = entries else {
            return;
        };

        let all_files = Self::collect_files_from_manifest_entries(entries, root, resource_type);
        let patterns: Vec<String> = entries
            .iter()
            .filter(|entry| is_override_pattern(entry))
            .cloned()
            .collect();
        let enabled_paths = apply_patterns(&all_files, &patterns, root);

        for f in &all_files {
            if enabled_paths.contains(f) {
                Self::add_resource(accumulator.target(resource_type), f, metadata, true);
            }
        }
    }

    /// The manifest entries' files, upstream's
    /// `collectFilesFromManifestEntries`: plain entries resolve against
    /// the root, glob entries expand (dot segments drop).
    fn collect_files_from_manifest_entries(
        entries: &[String],
        root: &str,
        resource_type: ResourceType,
    ) -> Vec<String> {
        let source_entries: Vec<String> = entries
            .iter()
            .filter(|entry| !is_override_pattern(entry))
            .cloned()
            .collect();
        let resolved = source_entries
            .iter()
            .flat_map(|entry| {
                if has_glob_pattern(entry) {
                    expand_package_glob(entry, root)
                } else {
                    vec![Path::new(root).join(entry).to_string_lossy().into_owned()]
                }
            })
            .collect::<Vec<_>>();
        Self::collect_files_from_paths(&resolved, resource_type)
    }

    /// The files behind a list of plain paths, upstream's
    /// `collectFilesFromPaths`: files pass through, directories collect
    /// their resource files.
    fn collect_files_from_paths(paths: &[String], resource_type: ResourceType) -> Vec<String> {
        let mut files = Vec::new();
        for p in paths {
            let Ok(fs_meta) = std::fs::metadata(p) else {
                continue;
            };
            if fs_meta.is_file() {
                files.push(p.clone());
            } else if fs_meta.is_dir() {
                files.extend(collect_resource_files(p, resource_type));
            }
        }
        files
    }

    /// Apply one settings entry list to a family, upstream's
    /// `resolveLocalEntries`: plain entries resolve against the base and
    /// collect, patterns then decide what stays enabled.
    fn resolve_local_entries(
        &self,
        entries: &[String],
        resource_type: ResourceType,
        accumulator: &mut ResourceAccumulator,
        metadata: &PathMetadata,
        base_dir: &str,
    ) {
        if entries.is_empty() {
            return;
        }

        let (plain, patterns) = split_patterns(entries);
        let resolved_plain: Vec<String> = plain
            .iter()
            .map(|p| {
                self.resolve_path_from_base(p, base_dir)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        let all_files = Self::collect_files_from_paths(&resolved_plain, resource_type);
        let enabled_paths = apply_patterns(&all_files, &patterns, base_dir);

        for f in &all_files {
            Self::add_resource(
                accumulator.target(resource_type),
                f,
                metadata,
                enabled_paths.contains(f),
            );
        }
    }

    /// Add the auto-discovered directories, upstream's
    /// `addAutoDiscoveredResources`.
    ///
    /// The project `.pi` tree (trust-gated), the project and user
    /// `.agents/skills` trees, and the user agent dir's four resource
    /// directories, each filtered by its settings override patterns.
    #[expect(
        clippy::too_many_lines,
        reason = "the 1:1 restatement of upstream's addAutoDiscoveredResources reads as one arm per source directory"
    )]
    fn add_auto_discovered_resources(
        &self,
        accumulator: &mut ResourceAccumulator,
        global_settings: &Settings,
        project_settings: &Settings,
        project_trusted: bool,
        global_base_dir: &str,
        project_base_dir: &str,
    ) {
        let user_metadata = PathMetadata {
            source: "auto".to_string(),
            scope: SourceScope::User,
            origin: SourceOrigin::TopLevel,
            base_dir: Some(global_base_dir.to_string()),
        };
        let project_metadata = PathMetadata {
            source: "auto".to_string(),
            scope: SourceScope::Project,
            origin: SourceOrigin::TopLevel,
            base_dir: Some(project_base_dir.to_string()),
        };

        let user_overrides = OverrideSets {
            extensions: settings_strings(global_settings, "extensions"),
            skills: settings_strings(global_settings, "skills"),
            prompts: settings_strings(global_settings, "prompts"),
            themes: settings_strings(global_settings, "themes"),
        };
        let project_overrides = OverrideSets {
            extensions: settings_strings(project_settings, "extensions"),
            skills: settings_strings(project_settings, "skills"),
            prompts: settings_strings(project_settings, "prompts"),
            themes: settings_strings(project_settings, "themes"),
        };

        let user_dirs = type_dirs(global_base_dir);
        let project_dirs = type_dirs(project_base_dir);
        let home = home_dir();
        let user_agents_skills_dir = Path::new(&home)
            .join(".agents")
            .join("skills")
            .to_string_lossy()
            .into_owned();
        let project_agents_skill_dirs: Vec<String> = if project_trusted {
            collect_ancestor_agents_skill_dirs(&self.cwd)
                .into_iter()
                .filter(|dir| resolve_default(dir) != resolve_default(&user_agents_skills_dir))
                .collect()
        } else {
            Vec::new()
        };

        let add_resources = |resource_type: ResourceType,
                             paths: &[String],
                             metadata: &PathMetadata,
                             overrides: &[String],
                             base_dir: &str,
                             accumulator: &mut ResourceAccumulator| {
            let target = accumulator.target(resource_type);
            for path in paths {
                let enabled = is_enabled_by_overrides(path, overrides, base_dir);
                Self::add_resource(target, path, metadata, enabled);
            }
        };

        if project_trusted {
            add_resources(
                ResourceType::Extensions,
                &collect_auto_extension_entries(&project_dirs.extensions),
                &project_metadata,
                &project_overrides.extensions,
                project_base_dir,
                accumulator,
            );

            add_resources(
                ResourceType::Skills,
                &collect_auto_skill_entries(&project_dirs.skills, SkillDiscoveryMode::Pi),
                &project_metadata,
                &project_overrides.skills,
                project_base_dir,
                accumulator,
            );
        }

        // Project skills from .agents/ trees, each with its own baseDir.
        for agents_skills_dir in &project_agents_skill_dirs {
            let agents_base_dir = dirname_posix(agents_skills_dir);
            let agents_metadata = PathMetadata {
                base_dir: Some(agents_base_dir.clone()),
                ..project_metadata.clone()
            };
            add_resources(
                ResourceType::Skills,
                &collect_auto_skill_entries(agents_skills_dir, SkillDiscoveryMode::Agents),
                &agents_metadata,
                &project_overrides.skills,
                &agents_base_dir,
                accumulator,
            );
        }

        if project_trusted {
            add_resources(
                ResourceType::Prompts,
                &collect_auto_prompt_entries(&project_dirs.prompts),
                &project_metadata,
                &project_overrides.prompts,
                project_base_dir,
                accumulator,
            );
            add_resources(
                ResourceType::Themes,
                &collect_auto_theme_entries(&project_dirs.themes),
                &project_metadata,
                &project_overrides.themes,
                project_base_dir,
                accumulator,
            );
        }

        add_resources(
            ResourceType::Extensions,
            &collect_auto_extension_entries(&user_dirs.extensions),
            &user_metadata,
            &user_overrides.extensions,
            global_base_dir,
            accumulator,
        );

        add_resources(
            ResourceType::Skills,
            &collect_auto_skill_entries(&user_dirs.skills, SkillDiscoveryMode::Pi),
            &user_metadata,
            &user_overrides.skills,
            global_base_dir,
            accumulator,
        );

        // User skills from ~/.agents/skills, with its own baseDir.
        let user_agents_base_dir = dirname_posix(&user_agents_skills_dir);
        let user_agents_metadata = PathMetadata {
            base_dir: Some(user_agents_base_dir.clone()),
            ..user_metadata.clone()
        };
        add_resources(
            ResourceType::Skills,
            &collect_auto_skill_entries(&user_agents_skills_dir, SkillDiscoveryMode::Agents),
            &user_agents_metadata,
            &user_overrides.skills,
            &user_agents_base_dir,
            accumulator,
        );

        add_resources(
            ResourceType::Prompts,
            &collect_auto_prompt_entries(&user_dirs.prompts),
            &user_metadata,
            &user_overrides.prompts,
            global_base_dir,
            accumulator,
        );
        add_resources(
            ResourceType::Themes,
            &collect_auto_theme_entries(&user_dirs.themes),
            &user_metadata,
            &user_overrides.themes,
            global_base_dir,
            accumulator,
        );
    }

    /// Record one resource path when unseen, upstream's `addResource`.
    fn add_resource(map: &mut TargetMap, path: &str, metadata: &PathMetadata, enabled: bool) {
        if path.is_empty() {
            return;
        }
        map.entry(path.to_string())
            .or_insert_with(|| (metadata.clone(), enabled));
    }

    /// The resource-path resolver, upstream's `resolvePath` with its
    /// default options: cwd base, tilde expansion, whitespace trim.
    #[must_use]
    #[expect(
        clippy::unused_self,
        reason = "the method mirrors upstream's private resolvePathFromBase on the manager"
    )]
    fn resolve_path_from_base(&self, input: &str, base_dir: &str) -> std::path::PathBuf {
        let resolved = resolve_path_with(
            input,
            base_dir,
            &PathInputOptions {
                trim: true,
                home_dir: Some(home_dir()),
                ..PathInputOptions::default()
            },
        );
        std::path::PathBuf::from(resolved.unwrap_or_else(|_| input.to_string()))
    }
}

/// The per-type directory names under one base, upstream's
/// `userDirs`/`projectDirs` objects.
fn type_dirs(base: &str) -> TypeDirs {
    TypeDirs {
        extensions: Path::new(base)
            .join("extensions")
            .to_string_lossy()
            .into_owned(),
        skills: Path::new(base)
            .join("skills")
            .to_string_lossy()
            .into_owned(),
        prompts: Path::new(base)
            .join("prompts")
            .to_string_lossy()
            .into_owned(),
        themes: Path::new(base)
            .join("themes")
            .to_string_lossy()
            .into_owned(),
    }
}

/// The four resource directories under one base directory.
struct TypeDirs {
    extensions: String,
    skills: String,
    prompts: String,
    themes: String,
}

/// The settings override arrays per family.
struct OverrideSets {
    extensions: Vec<String>,
    skills: Vec<String>,
    prompts: Vec<String>,
    themes: Vec<String>,
}

/// The string entries behind one settings key, upstream's
/// `(settings[key] ?? []) as string[]` — non-string entries drop, the way
/// the collectors would choke on them.
fn settings_strings(settings: &Settings, key: &str) -> Vec<String> {
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

/// The settings `packages` entries as (source, scope) pairs, upstream's
/// `allPackages` collection: project scope first so project resources win
/// collisions, string and object entries both reading their `source`.
fn package_sources(settings: &Settings, scope: SourceScope) -> Vec<(String, SourceScope)> {
    settings
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .map(|entry| (package_source_string(entry), scope))
                .collect()
        })
        .unwrap_or_default()
}

/// One settings `packages` entry's source string, upstream's
/// `getPackageSourceString`.
fn package_source_string(entry: &serde_json::Value) -> String {
    match entry {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(object) => object
            .get("source")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

/// Assemble the resolved paths, upstream's `toResolvedPaths`: sort by
/// precedence rank (stable, so same-rank insertion order survives) and
/// drop canonical duplicates, first wins.
fn to_resolved_paths(accumulator: ResourceAccumulator) -> ResolvedPaths {
    fn map_to_resolved(entries: TargetMap) -> Vec<ResolvedResource> {
        let mut resolved: Vec<ResolvedResource> = entries
            .into_iter()
            .map(|(path, (metadata, enabled))| ResolvedResource {
                path,
                enabled,
                metadata,
            })
            .collect();
        resolved.sort_by_key(|entry| resource_precedence_rank(&entry.metadata));

        let mut seen: HashSet<String> = HashSet::new();
        resolved
            .into_iter()
            .filter(|entry| seen.insert(canonicalize_path(&entry.path)))
            .collect()
    }

    ResolvedPaths {
        extensions: map_to_resolved(accumulator.extensions),
        skills: map_to_resolved(accumulator.skills),
        prompts: map_to_resolved(accumulator.prompts),
        themes: map_to_resolved(accumulator.themes),
    }
}

/// The resolver defaults, upstream's `resolvePath(input)` with no base:
/// the process cwd is the base and the process home expands tildes.
#[must_use]
fn resolve_default(input: &str) -> String {
    let base = crate::config::process_cwd();
    resolve_path_with(
        input,
        &base,
        &PathInputOptions {
            home_dir: Some(home_dir()),
            ..PathInputOptions::default()
        },
    )
    .unwrap_or_else(|_| input.to_string())
}

/// The parent of a POSIX path, `None` at the root — the walk terminators
/// upstream's `dirname(dir) === dir` checks express.
#[must_use]
fn parent_dir(dir: &str) -> Option<String> {
    let trimmed = dir.trim_end_matches('/');
    let parent = match trimmed.rfind('/') {
        Some(0) => "/",
        Some(at) => &trimmed[..at],
        None => return None,
    };
    if parent == trimmed {
        return None;
    }
    Some(parent.to_string())
}

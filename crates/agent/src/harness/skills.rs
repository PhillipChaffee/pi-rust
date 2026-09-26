//! The skill loader, ported from upstream `src/harness/skills.ts`.
//!
//! `loadSkills` walks one or more directories through the
//! [`ExecutionEnv`] capability: each
//! directory loads at most one `SKILL.md` (the first in listing order), root
//! `.md` children load only at the traversal root, and `.gitignore` /
//! `.ignore` / `.fdignore` files filter the walk through one matcher per
//! top-level directory. Missing input directories are skipped.
//!
//! The ignore matcher restates upstream's npm `ignore` dependency on the
//! Rust `ignore` crate: the builder runs case-insensitively, the way npm
//! `ignore` defaults, and [`IgnoreMatcher::ignores`] re-expresses npm's
//! parent-first walk (an ignored parent directory ignores everything below
//! it; whitelists cannot re-include under an ignored parent) over
//! `Gitignore::matched`, which checks one path at a time.

use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};

use crate::harness::context::Context;
use crate::harness::fs_scan::{
    LoadDiagnostic, SourcedDiagnostic, SourcedPath, dirname_env_path, frontmatter_is_true,
    frontmatter_string, parse_frontmatter, relative_env_path, resolve_kind, sort_entries_by_name,
};
use crate::harness::types::{ExecutionEnv, FileErrorCode, FileInfo, FileKind, Skill};

/// The stable skill-loader diagnostic codes, upstream's `SkillDiagnosticCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkillDiagnosticCode {
    /// A file-info or canonical-path probe failed outside `not_found`.
    FileInfoFailed,
    /// Listing a directory failed.
    ListFailed,
    /// Reading a file failed.
    ReadFailed,
    /// The frontmatter of a declared skill file does not parse.
    ParseFailed,
    /// The skill metadata fails validation.
    InvalidMetadata,
}

/// A warning produced while loading skills, upstream's `SkillDiagnostic`.
pub type SkillDiagnostic = LoadDiagnostic<SkillDiagnosticCode>;

/// Longest accepted skill name, upstream's `MAX_NAME_LENGTH`.
const MAX_NAME_LENGTH: usize = 64;

/// Longest accepted skill description, upstream's `MAX_DESCRIPTION_LENGTH`.
const MAX_DESCRIPTION_LENGTH: usize = 1024;

/// The ignore files each traversed directory contributes to the matcher,
/// upstream's `IGNORE_FILE_NAMES`.
const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

/// Format a skill invocation prompt, optionally appending additional user
/// instructions, upstream's `formatSkillInvocation`.
///
/// The block names the skill and its file and points relative references at
/// the skill file's directory.
#[must_use]
pub fn format_skill_invocation(skill: &Skill, additional_instructions: Option<&str>) -> String {
    let skill_block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path,
        dirname_env_path(&skill.file_path),
        skill.content
    );
    match additional_instructions {
        Some(instructions) => format!("{skill_block}\n\n{instructions}"),
        None => skill_block,
    }
}

/// The skills and diagnostics one load produced, upstream's
/// `{ skills, diagnostics }` return object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadedSkills {
    /// The skills that loaded, in traversal order.
    pub skills: Vec<Skill>,
    /// The warnings the load produced.
    pub diagnostics: Vec<SkillDiagnostic>,
}

/// A skill paired with the source it was loaded for, upstream's
/// `{ skill, source }` element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourcedSkill<TSkill, TSource> {
    /// The skill, mapped when the caller supplied a mapper.
    pub skill: TSkill,
    /// The source value the loader input carried.
    pub source: TSource,
}

/// The sourced skills and diagnostics one load produced, upstream's
/// `{ skills, diagnostics }` return object of `loadSourcedSkills`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedSourcedSkills<TSkill, TSource> {
    /// The sourced skills, in input then traversal order.
    pub skills: Vec<SourcedSkill<TSkill, TSource>>,
    /// The warnings with their sources attached.
    pub diagnostics: Vec<SourcedDiagnostic<SkillDiagnostic, TSource>>,
}

/// The optional skill mapper, upstream's `mapSkill` parameter: given the
/// loaded skill, the input's source, and the context, it produces the
/// application's skill type.
pub type SkillMapper<'map, TSource, TSkill> = &'map dyn Fn(&Skill, &TSource, &Context) -> TSkill;

/// Load skills from one or more directories, upstream's `loadSkills`.
///
/// Traverses directories recursively, loads `SKILL.md` files, loads direct
/// root `.md` files with skill frontmatter, honors ignore files, and returns
/// diagnostics for invalid declared skill files. Missing input directories
/// are skipped.
pub async fn load_skills(
    env: &dyn ExecutionEnv,
    dirs: &[String],
    context: &Context,
) -> LoadedSkills {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    for dir in dirs {
        let root_info = match env.file_info(dir, context).await {
            Err(error) => {
                if error.code != FileErrorCode::NotFound {
                    diagnostics.push(SkillDiagnostic::warning(
                        SkillDiagnosticCode::FileInfoFailed,
                        error.message,
                        dir.clone(),
                    ));
                }
                continue;
            }
            Ok(root_info) => root_info,
        };
        if resolve_kind(
            env,
            &root_info,
            &mut diagnostics,
            SkillDiagnosticCode::FileInfoFailed,
            context,
        )
        .await
            != Some(FileKind::Directory)
        {
            continue;
        }
        let mut matcher = IgnoreMatcher::new();
        let (loaded, loaded_diagnostics) = load_skills_from_dir_internal(
            env,
            &root_info.path,
            true,
            &mut matcher,
            &root_info.path,
            context,
        )
        .await;
        skills.extend(loaded);
        diagnostics.extend(loaded_diagnostics);
    }
    LoadedSkills {
        skills,
        diagnostics,
    }
}

/// Load skills from source-tagged directories, upstream's `loadSourcedSkills`
/// with no mapper.
///
/// Source values are preserved exactly and attached to every loaded skill
/// and diagnostic. The agent package does not interpret source values;
/// applications define their own provenance shape. Upstream's optional
/// mapper and default `TSkill` restates as the identity variant here and
/// [`load_sourced_skills_mapped`] beside it: TypeScript's `TSkill extends
/// Skill` has no structural-subtyping equivalent, so the identity case
/// clones through a pass-through mapper.
pub async fn load_sourced_skills<TSource: Clone>(
    env: &dyn ExecutionEnv,
    inputs: &[SourcedPath<TSource>],
    context: &Context,
) -> LoadedSourcedSkills<Skill, TSource> {
    load_sourced_skills_mapped(
        env,
        inputs,
        &|skill, _source, _context| skill.clone(),
        context,
    )
    .await
}

/// Load skills from source-tagged directories with a mapper, upstream's
/// `loadSourcedSkills` with `mapSkill`.
///
/// The mapper receives the loaded skill, the input's source, and the
/// context, and produces the application's skill type.
pub async fn load_sourced_skills_mapped<TSource: Clone, TSkill>(
    env: &dyn ExecutionEnv,
    inputs: &[SourcedPath<TSource>],
    map_skill: SkillMapper<'_, TSource, TSkill>,
    context: &Context,
) -> LoadedSourcedSkills<TSkill, TSource> {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    for input in inputs {
        let result = load_skills(env, std::slice::from_ref(&input.path), context).await;
        for skill in result.skills {
            skills.push(SourcedSkill {
                skill: map_skill(&skill, &input.source, context),
                source: input.source.clone(),
            });
        }
        for diagnostic in result.diagnostics {
            diagnostics.push(SourcedDiagnostic {
                diagnostic,
                source: input.source.clone(),
            });
        }
    }
    LoadedSourcedSkills {
        skills,
        diagnostics,
    }
}

/// Walk one directory, upstream's `loadSkillsFromDirInternal`.
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's `entry.name.endsWith(\".md\")` is case-sensitive; the 1:1 shape keeps it"
)]
async fn load_skills_from_dir_internal(
    env: &dyn ExecutionEnv,
    dir: &str,
    include_root_files: bool,
    matcher: &mut IgnoreMatcher,
    root_dir: &str,
    context: &Context,
) -> (Vec<Skill>, Vec<SkillDiagnostic>) {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    let file_info_failed = SkillDiagnosticCode::FileInfoFailed;

    let dir_info = match env.file_info(dir, context).await {
        Err(error) => {
            if error.code != FileErrorCode::NotFound {
                diagnostics.push(SkillDiagnostic::warning(
                    file_info_failed,
                    error.message,
                    dir,
                ));
            }
            return (skills, diagnostics);
        }
        Ok(dir_info) => dir_info,
    };
    if resolve_kind(env, &dir_info, &mut diagnostics, file_info_failed, context).await
        != Some(FileKind::Directory)
    {
        return (skills, diagnostics);
    }

    add_ignore_rules(env, matcher, dir, root_dir, &mut diagnostics, context).await;

    let mut entries = match env.list_dir(dir, context).await {
        Err(error) => {
            diagnostics.push(SkillDiagnostic::warning(
                SkillDiagnosticCode::ListFailed,
                error.message,
                dir,
            ));
            return (skills, diagnostics);
        }
        Ok(entries) => entries,
    };

    // The first SKILL.md in listing order owns the directory: loading it
    // returns immediately, so neither sibling root files nor subdirectories
    // of this directory load, upstream's in-loop `return`.
    let (stop, skill) = load_first_skill_md(
        env,
        &entries,
        matcher,
        root_dir,
        &dir_info.name,
        &mut diagnostics,
        context,
    )
    .await;
    skills.extend(skill);
    if stop {
        return (skills, diagnostics);
    }

    sort_entries_by_name(&mut entries);
    for entry in &entries {
        if entry.name.starts_with('.') || entry.name == "node_modules" {
            continue;
        }
        let Some(kind) =
            resolve_kind(env, entry, &mut diagnostics, file_info_failed, context).await
        else {
            continue;
        };
        let rel_path = relative_env_path(root_dir, &entry.path);
        let ignore_path = if kind == FileKind::Directory {
            format!("{rel_path}/")
        } else {
            rel_path.clone()
        };
        if matcher.ignores(&ignore_path) {
            continue;
        }

        if kind == FileKind::Directory {
            let (loaded, loaded_diagnostics) = Box::pin(load_skills_from_dir_internal(
                env,
                &entry.path,
                false,
                matcher,
                root_dir,
                context,
            ))
            .await;
            skills.extend(loaded);
            diagnostics.extend(loaded_diagnostics);
            continue;
        }

        if kind != FileKind::File || !include_root_files || !entry.name.ends_with(".md") {
            continue;
        }
        let (skill, loaded_diagnostics) =
            load_skill_from_file(env, &entry.path, &dir_info.name, context).await;
        if let Some(skill) = skill {
            skills.push(skill);
        }
        diagnostics.extend(loaded_diagnostics);
    }

    (skills, diagnostics)
}

/// Load the first listed `SKILL.md`, upstream's first entry loop.
///
/// Returns whether the walk must stop — one `SKILL.md` was attempted,
/// whether or not it loaded (upstream's in-loop `return`) — with the loaded
/// skill; diagnostics flow through the shared vector.
async fn load_first_skill_md(
    env: &dyn ExecutionEnv,
    entries: &[FileInfo],
    matcher: &IgnoreMatcher,
    root_dir: &str,
    parent_dir_name: &str,
    diagnostics: &mut Vec<SkillDiagnostic>,
    context: &Context,
) -> (bool, Option<Skill>) {
    for entry in entries {
        if entry.name != "SKILL.md" {
            continue;
        }
        if resolve_kind(
            env,
            entry,
            diagnostics,
            SkillDiagnosticCode::FileInfoFailed,
            context,
        )
        .await
            != Some(FileKind::File)
        {
            continue;
        }
        let rel_path = relative_env_path(root_dir, &entry.path);
        if matcher.ignores(&rel_path) {
            continue;
        }
        let (skill, loaded_diagnostics) =
            load_skill_from_file(env, &entry.path, parent_dir_name, context).await;
        diagnostics.extend(loaded_diagnostics);
        return (true, skill);
    }
    (false, None)
}

/// Add one directory's ignore-file patterns to the matcher, upstream's
/// `addIgnoreRules`.
///
/// Patterns prefix with the directory's path relative to the traversal root
/// so a subdirectory's ignore file applies at its own depth; `not_found`
/// probes are silent, everything else warns.
async fn add_ignore_rules(
    env: &dyn ExecutionEnv,
    matcher: &mut IgnoreMatcher,
    dir: &str,
    root_dir: &str,
    diagnostics: &mut Vec<SkillDiagnostic>,
    context: &Context,
) {
    let relative_dir = relative_env_path(root_dir, dir);
    let prefix = if relative_dir.is_empty() {
        String::new()
    } else {
        format!("{relative_dir}/")
    };

    for filename in IGNORE_FILE_NAMES {
        let parts = [dir.to_owned(), filename.to_owned()];
        let ignore_path = match env.join_path(&parts, context).await {
            Err(error) => {
                diagnostics.push(SkillDiagnostic::warning(
                    SkillDiagnosticCode::FileInfoFailed,
                    error.message,
                    dir,
                ));
                continue;
            }
            Ok(ignore_path) => ignore_path,
        };
        let info = match env.file_info(&ignore_path, context).await {
            Err(error) => {
                if error.code != FileErrorCode::NotFound {
                    diagnostics.push(SkillDiagnostic::warning(
                        SkillDiagnosticCode::FileInfoFailed,
                        error.message,
                        ignore_path.clone(),
                    ));
                }
                continue;
            }
            Ok(info) => info,
        };
        if info.kind != FileKind::File {
            continue;
        }
        let ignore_content = match env.read_text_file(&ignore_path, context).await {
            Err(error) => {
                diagnostics.push(SkillDiagnostic::warning(
                    SkillDiagnosticCode::ReadFailed,
                    error.message,
                    ignore_path,
                ));
                continue;
            }
            Ok(ignore_content) => ignore_content,
        };
        let patterns: Vec<String> = ignore_content
            .split('\n')
            .map(|line| line.strip_suffix('\r').unwrap_or(line))
            .filter_map(|line| prefix_ignore_pattern(line, &prefix))
            .collect();
        if !patterns.is_empty() {
            matcher.add(&patterns);
        }
    }
}

/// Rewrite one ignore-file line so it applies relative to the traversal
/// root, upstream's `prefixIgnorePattern`.
///
/// Blank lines and comments (an unescaped leading `#`) drop. A leading `!`
/// negates; an escaped `\!` keeps its bang through the prefixing, which the
/// matcher then reads as a negation of the unprefixed pattern, as upstream's
/// handoff to npm `ignore` does. A leading `/` drops before the prefix,
/// preserving the root-anchored reading; unanchored patterns stay
/// unanchored at their new depth.
fn prefix_ignore_pattern(line: &str, prefix: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('#') && !trimmed.starts_with("\\#") {
        return None;
    }

    let mut pattern = line;
    let mut negated = false;
    if pattern.starts_with('!') {
        negated = true;
        pattern = &pattern[1..];
    } else if let Some(stripped) = pattern.strip_prefix("\\!") {
        pattern = stripped;
    }
    if let Some(stripped) = pattern.strip_prefix('/') {
        pattern = stripped;
    }
    let prefixed = if prefix.is_empty() {
        pattern.to_owned()
    } else {
        format!("{prefix}{pattern}")
    };
    if negated {
        Some(format!("!{prefixed}"))
    } else {
        Some(prefixed)
    }
}

/// Load one skill file, upstream's `loadSkillFromFile`.
///
/// Declared skill files (`SKILL.md`) report parse and metadata warnings;
/// root `.md` files stay silent on parse failure and a missing description
/// but still carry metadata warnings when a description loads.
async fn load_skill_from_file(
    env: &dyn ExecutionEnv,
    file_path: &str,
    parent_dir_name: &str,
    context: &Context,
) -> (Option<Skill>, Vec<SkillDiagnostic>) {
    let mut diagnostics = Vec::new();
    let is_declared_skill = file_path
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        == Some("SKILL.md");
    let raw_content = match env.read_text_file(file_path, context).await {
        Err(error) => {
            diagnostics.push(SkillDiagnostic::warning(
                SkillDiagnosticCode::ReadFailed,
                error.message,
                file_path,
            ));
            return (None, diagnostics);
        }
        Ok(raw_content) => raw_content,
    };

    let parsed = match parse_frontmatter(&raw_content) {
        Err(error) => {
            if is_declared_skill {
                diagnostics.push(SkillDiagnostic::warning(
                    SkillDiagnosticCode::ParseFailed,
                    error.0,
                    file_path,
                ));
            }
            return (None, diagnostics);
        }
        Ok(parsed) => parsed,
    };

    let description = frontmatter_string(&parsed.frontmatter, "description").map(str::to_owned);
    if !is_declared_skill
        && description
            .as_ref()
            .is_none_or(|description| description.trim().is_empty())
    {
        return (None, diagnostics);
    }

    for error in validate_description(description.as_deref()) {
        diagnostics.push(SkillDiagnostic::warning(
            SkillDiagnosticCode::InvalidMetadata,
            error,
            file_path,
        ));
    }

    let frontmatter_name = frontmatter_string(&parsed.frontmatter, "name").map(str::to_owned);
    let name = frontmatter_name
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| parent_dir_name.to_owned());
    for error in validate_name(&name, parent_dir_name) {
        diagnostics.push(SkillDiagnostic::warning(
            SkillDiagnosticCode::InvalidMetadata,
            error,
            file_path,
        ));
    }

    let Some(description) = description.filter(|description| !description.trim().is_empty()) else {
        return (None, diagnostics);
    };

    (
        Some(Skill {
            name,
            description,
            content: parsed.body,
            file_path: file_path.to_owned(),
            disable_model_invocation: Some(frontmatter_is_true(
                &parsed.frontmatter,
                "disable-model-invocation",
            )),
        }),
        diagnostics,
    )
}

/// Validate a skill name against the declared rules, upstream's
/// `validateName`.
///
/// Length counts characters, upstream's `.length` UTF-16 units restated as
/// Rust scalar counts — identical on the BMP names skills carry.
fn validate_name(name: &str, parent_dir_name: &str) -> Vec<String> {
    let mut errors = Vec::new();
    if name != parent_dir_name {
        errors.push(format!(
            "name \"{name}\" does not match parent directory \"{parent_dir_name}\""
        ));
    }
    if name.chars().count() > MAX_NAME_LENGTH {
        errors.push(format!(
            "name exceeds {MAX_NAME_LENGTH} characters ({})",
            name.chars().count()
        ));
    }
    let characters_ok = !name.is_empty()
        && name.chars().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
        });
    if !characters_ok {
        errors.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_owned(),
        );
    }
    if name.starts_with('-') || name.ends_with('-') {
        errors.push("name must not start or end with a hyphen".to_owned());
    }
    if name.contains("--") {
        errors.push("name must not contain consecutive hyphens".to_owned());
    }
    errors
}

/// Validate a skill description, upstream's `validateDescription`.
fn validate_description(description: Option<&str>) -> Vec<String> {
    match description {
        None => vec!["description is required".to_owned()],
        Some(description) if description.trim().is_empty() => {
            vec!["description is required".to_owned()]
        }
        Some(description) if description.chars().count() > MAX_DESCRIPTION_LENGTH => vec![format!(
            "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({})",
            description.chars().count()
        )],
        Some(_) => Vec::new(),
    }
}

/// The per-directory ignore matcher, upstream's `IgnoreMatcher`
/// (`ReturnType<typeof ignore>`) built on the Rust `ignore` crate.
struct IgnoreMatcher {
    builder: GitignoreBuilder,
    matcher: Gitignore,
}

impl IgnoreMatcher {
    fn new() -> Self {
        let mut builder = GitignoreBuilder::new("");
        // npm `ignore` matches case-insensitively by default; the builder
        // re-translates no lines yet, so this cannot fail.
        let _ = builder.case_insensitive(true);
        Self {
            matcher: Gitignore::empty(),
            builder,
        }
    }

    /// Add pattern lines, upstream's `ig.add(patterns)`.
    ///
    /// npm filters blank, comment, and dangling-backslash lines rather than
    /// rejecting them; a line the builder refuses drops the same way.
    fn add(&mut self, patterns: &[String]) {
        if patterns.is_empty() {
            return;
        }
        let mut added = false;
        for pattern in patterns {
            if self.builder.add_line(None, pattern).is_ok() {
                added = true;
            }
        }
        if added && let Ok(matcher) = self.builder.build() {
            self.matcher = matcher;
        }
    }

    /// Whether the relative path is ignored, upstream's `ig.ignores(relPath)`
    /// (a directory's relative path carries a trailing `/`).
    ///
    /// npm `ignore` walks parent-first: an ignored parent directory ignores
    /// the path outright and its own whitelist patterns never rescue it, and
    /// a whitelist match only overrides ignores matched at the same level.
    /// The walk re-expresses that over `Gitignore::matched`; ancestors test
    /// as directories.
    fn ignores(&self, rel_path: &str) -> bool {
        let is_dir = rel_path.ends_with('/');
        let path = rel_path.strip_suffix('/').unwrap_or(rel_path);
        let segments: Vec<&str> = path
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect();
        for depth in 1..segments.len() {
            let ancestor = segments[..depth].join("/");
            if matches!(self.matcher.matched(&ancestor, true), Match::Ignore(_)) {
                return true;
            }
        }
        matches!(self.matcher.matched(path, is_dir), Match::Ignore(_))
    }
}

#[cfg(test)]
mod tests;

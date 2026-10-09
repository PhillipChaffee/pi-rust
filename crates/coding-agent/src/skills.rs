//! The skill loader, upstream's `src/core/skills.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! This is the coding-agent's own loader — a sibling of pi-agent-core's
//! harness loader (#95) with a different surface: provenance-carrying
//! skills, the Agent-Skills-spec name/description validation without the
//! parent-directory-match rule, collision diagnostics, and the XML
//! `<available_skills>` prompt block. Discovery is the same machinery the
//! package manager's skill collector re-implements upstream
//! (`SKILL.md`-first, one per directory, root `.md` per the flavor), and
//! the ignore rules restate npm `ignore` through the workspace's single
//! matcher restatement, `pi_agent_core::harness::skills`.

use std::path::Path;

use indexmap::IndexMap;

use pi_agent_core::harness::skills::{IgnoreMatcher, prefix_ignore_pattern};

use crate::diagnostics::{
    ResourceCollision, ResourceDiagnostic, ResourceDiagnosticKind, ResourceType,
};
use crate::source_info::{SourceInfo, SyntheticSourceOptions, create_synthetic_source_info};
use crate::utils::frontmatter::{
    ParsedFrontmatter, frontmatter_is_true, frontmatter_string, parse_frontmatter,
};
use crate::utils::paths::{
    PathInputOptions, basename_posix, canonicalize_path, dirname_posix, is_under_path,
    relative_posix, resolve_path, resolve_path_with,
};

/// Longest accepted skill name, upstream's `MAX_NAME_LENGTH`.
const MAX_NAME_LENGTH: usize = 64;

/// Longest accepted skill description, upstream's `MAX_DESCRIPTION_LENGTH`.
const MAX_DESCRIPTION_LENGTH: usize = 1024;

/// The ignore files each traversed directory contributes, upstream's
/// `IGNORE_FILE_NAMES`.
const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

/// A loaded skill, upstream's `Skill`: the prompt-visible identity, the
/// file it loaded from, and its provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    /// The skill name — the frontmatter `name` or the parent directory's
    /// name.
    pub name: String,
    /// The frontmatter description.
    pub description: String,
    /// The absolute path of the file the skill loaded from.
    pub file_path: String,
    /// The skill's directory, the base relative references resolve
    /// against.
    pub base_dir: String,
    /// Where the skill came from.
    pub source_info: SourceInfo,
    /// Whether the model may invoke the skill unprompted, upstream's
    /// `disableModelInvocation`.
    pub disable_model_invocation: bool,
}

/// The skills and diagnostics one load produced, upstream's
/// `LoadSkillsResult`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadSkillsResult {
    /// The skills that loaded, in discovery order.
    pub skills: Vec<Skill>,
    /// The warnings and collisions the load produced.
    pub diagnostics: Vec<ResourceDiagnostic>,
}

/// A skill frontmatter document, upstream's `SkillFrontmatter`.
#[derive(Debug, Default)]
pub struct SkillFrontmatter {
    /// The declared name.
    pub name: Option<String>,
    /// The declared description.
    pub description: Option<String>,
    /// Whether the skill stays out of the model-visible block.
    pub disable_model_invocation: bool,
}

/// Validate a skill name against the Agent Skills spec, upstream's
/// `validateName`. Returns the violation messages.
#[must_use]
fn validate_name(name: &str) -> Vec<String> {
    let mut errors = Vec::new();

    if name.chars().count() > MAX_NAME_LENGTH {
        errors.push(format!(
            "name exceeds {MAX_NAME_LENGTH} characters ({})",
            name.chars().count()
        ));
    }

    let characters_ok = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if !characters_ok {
        errors.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_string(),
        );
    }

    if name.starts_with('-') || name.ends_with('-') {
        errors.push("name must not start or end with a hyphen".to_string());
    }

    if name.contains("--") {
        errors.push("name must not contain consecutive hyphens".to_string());
    }

    errors
}

/// Validate a skill description against the Agent Skills spec, upstream's
/// `validateDescription`.
#[must_use]
fn validate_description(description: Option<&str>) -> Vec<String> {
    match description {
        None => vec!["description is required".to_string()],
        Some(description) if description.trim().is_empty() => {
            vec!["description is required".to_string()]
        }
        Some(description) if description.chars().count() > MAX_DESCRIPTION_LENGTH => vec![format!(
            "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({})",
            description.chars().count()
        )],
        Some(_) => Vec::new(),
    }
}

/// The provenance a skill's `source` tag carries, upstream's
/// `createSkillSourceInfo`: the user/project/path tags restate as local
/// sources at their scope, anything else passes through verbatim.
#[must_use]
fn create_skill_source_info(file_path: &str, base_dir: &str, source: &str) -> SourceInfo {
    match source {
        "user" => create_synthetic_source_info(
            file_path,
            &SyntheticSourceOptions {
                source: "local".to_string(),
                scope: Some(crate::source_info::SourceScope::User),
                origin: None,
                base_dir: Some(base_dir.to_string()),
            },
        ),
        "project" => create_synthetic_source_info(
            file_path,
            &SyntheticSourceOptions {
                source: "local".to_string(),
                scope: Some(crate::source_info::SourceScope::Project),
                origin: None,
                base_dir: Some(base_dir.to_string()),
            },
        ),
        "path" => create_synthetic_source_info(
            file_path,
            &SyntheticSourceOptions {
                source: "local".to_string(),
                scope: None,
                origin: None,
                base_dir: Some(base_dir.to_string()),
            },
        ),
        other => create_synthetic_source_info(
            file_path,
            &SyntheticSourceOptions {
                source: other.to_string(),
                scope: None,
                origin: None,
                base_dir: Some(base_dir.to_string()),
            },
        ),
    }
}

/// The directory-load options, upstream's `LoadSkillsFromDirOptions`.
#[derive(Debug)]
pub struct LoadSkillsFromDirOptions<'a> {
    /// The directory to scan for skills.
    pub dir: &'a str,
    /// The source identifier for these skills.
    pub source: &'a str,
}

/// Load skills from one directory, upstream's `loadSkillsFromDir`.
#[must_use]
pub fn load_skills_from_dir(options: &LoadSkillsFromDirOptions<'_>) -> LoadSkillsResult {
    let mut matcher = IgnoreMatcher::new();
    let (skills, diagnostics) =
        load_skills_from_dir_internal(options.dir, options.source, true, &mut matcher, options.dir);
    LoadSkillsResult {
        skills,
        diagnostics,
    }
}

/// Walk one directory, upstream's `loadSkillsFromDirInternal`.
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's md ends-with probe is case-sensitive; the 1:1 shape keeps it"
)]
fn load_skills_from_dir_internal(
    dir: &str,
    source: &str,
    include_root_files: bool,
    matcher: &mut IgnoreMatcher,
    root_dir: &str,
) -> (Vec<Skill>, Vec<ResourceDiagnostic>) {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();

    if !Path::new(dir).exists() {
        return (skills, diagnostics);
    }

    add_ignore_rules(matcher, dir, root_dir);

    let Ok(read) = std::fs::read_dir(dir) else {
        // Upstream wraps the whole walk in a bare try/catch.
        return (skills, diagnostics);
    };
    let entries: Vec<std::fs::DirEntry> = read.flatten().collect();

    // The first listed SKILL.md owns the directory: attempted or not, the
    // walk stops there, upstream's in-loop `return`.
    for entry in &entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != "SKILL.md" {
            continue;
        }

        let full_path = Path::new(dir).join(&name).to_string_lossy().into_owned();

        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let is_file = if file_type.is_symlink() {
            match crate::utils::paths::symlink_resolved_kind(Path::new(&full_path)) {
                Some((_, is_file)) => is_file,
                None => continue,
            }
        } else {
            file_type.is_file()
        };

        let rel_path = relative_posix(root_dir, &full_path);
        if !is_file || matcher.ignores(&rel_path) {
            continue;
        }

        let (skill, loaded_diagnostics) = load_skill_from_file(&full_path, source);
        if let Some(skill) = skill {
            skills.push(skill);
        }
        diagnostics.extend(loaded_diagnostics);
        return (skills, diagnostics);
    }

    for entry in &entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }

        // Skip node_modules to avoid scanning dependencies.
        if name == "node_modules" {
            continue;
        }

        let full_path = Path::new(dir).join(&name).to_string_lossy().into_owned();

        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let (is_directory, is_file) = if file_type.is_symlink() {
            match crate::utils::paths::symlink_resolved_kind(Path::new(&full_path)) {
                Some(kind) => kind,
                None => continue,
            }
        } else {
            (file_type.is_dir(), file_type.is_file())
        };

        let rel_path = relative_posix(root_dir, &full_path);
        let ignore_path = if is_directory {
            format!("{rel_path}/")
        } else {
            rel_path
        };
        if matcher.ignores(&ignore_path) {
            continue;
        }

        if is_directory {
            let (sub_skills, sub_diagnostics) =
                load_skills_from_dir_internal(&full_path, source, false, matcher, root_dir);
            skills.extend(sub_skills);
            diagnostics.extend(sub_diagnostics);
            continue;
        }

        if !is_file || !include_root_files || !name.ends_with(".md") {
            continue;
        }

        let (skill, loaded_diagnostics) = load_skill_from_file(&full_path, source);
        if let Some(skill) = skill {
            skills.push(skill);
        }
        diagnostics.extend(loaded_diagnostics);
    }

    (skills, diagnostics)
}

/// Add one directory's ignore-file patterns to the matcher, upstream's
/// `addIgnoreRules`.
fn add_ignore_rules(matcher: &mut IgnoreMatcher, dir: &str, root_dir: &str) {
    let relative_dir = relative_posix(root_dir, dir);
    let prefix = if relative_dir.is_empty() {
        String::new()
    } else {
        format!("{relative_dir}/")
    };

    for filename in IGNORE_FILE_NAMES {
        let Ok(content) = std::fs::read_to_string(Path::new(dir).join(filename)) else {
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

/// Load one skill file, upstream's `loadSkillFromFile`.
///
/// Declared skill files (`SKILL.md`) report parse and metadata warnings;
/// root `.md` files stay silent on parse failure and a missing
/// description but still carry metadata warnings when a description
/// loads. The skill loads even with warnings, unless the description is
/// missing or empty.
fn load_skill_from_file(file_path: &str, source: &str) -> (Option<Skill>, Vec<ResourceDiagnostic>) {
    let mut diagnostics = Vec::new();
    let is_declared_skill = basename_posix(file_path) == "SKILL.md";

    let Ok(raw_content) = std::fs::read_to_string(file_path) else {
        diagnostics.push(ResourceDiagnostic {
            kind: ResourceDiagnosticKind::Warning,
            message: format!("failed to read skill file: {file_path}"),
            path: Some(file_path.to_string()),
            collision: None,
        });
        return (None, diagnostics);
    };

    let parsed: ParsedFrontmatter = match parse_frontmatter(&raw_content) {
        Ok(parsed) => parsed,
        Err(error) => {
            if is_declared_skill {
                diagnostics.push(ResourceDiagnostic {
                    kind: ResourceDiagnosticKind::Warning,
                    message: error.0,
                    path: Some(file_path.to_string()),
                    collision: None,
                });
            }
            return (None, diagnostics);
        }
    };

    let description = frontmatter_string(&parsed.frontmatter, "description").map(str::to_string);
    let has_description = description.as_ref().is_some_and(|d| !d.trim().is_empty());
    if !is_declared_skill && !has_description {
        return (None, diagnostics);
    }

    let skill_dir = dirname_posix(file_path);
    let parent_dir_name = basename_posix(&skill_dir);

    for error in validate_description(description.as_deref()) {
        diagnostics.push(ResourceDiagnostic {
            kind: ResourceDiagnosticKind::Warning,
            message: error,
            path: Some(file_path.to_string()),
            collision: None,
        });
    }

    // An empty frontmatter name falls back to the parent directory's
    // name, upstream's `frontmatterName || parentDirName`.
    let frontmatter_name = frontmatter_string(&parsed.frontmatter, "name")
        .filter(|name| !name.is_empty())
        .map(str::to_string);
    let name = frontmatter_name.unwrap_or_else(|| parent_dir_name.clone());

    for error in validate_name(&name) {
        diagnostics.push(ResourceDiagnostic {
            kind: ResourceDiagnosticKind::Warning,
            message: error,
            path: Some(file_path.to_string()),
            collision: None,
        });
    }

    let Some(description) = description.filter(|d| !d.trim().is_empty()) else {
        return (None, diagnostics);
    };

    (
        Some(Skill {
            name,
            description,
            file_path: file_path.to_string(),
            base_dir: skill_dir.clone(),
            source_info: create_skill_source_info(file_path, &skill_dir, source),
            disable_model_invocation: frontmatter_is_true(
                &parsed.frontmatter,
                "disable-model-invocation",
            ),
        }),
        diagnostics,
    )
}

/// Format skills for the system prompt, upstream's
/// `formatSkillsForPrompt`.
///
/// The XML `<available_skills>` block per the Agent Skills integration
/// standard, with skills flagged `disable-model-invocation` left out. An
/// empty visible set formats as the empty string.
#[must_use]
pub fn format_skills_for_prompt(skills: &[Skill], file_read_tool: &str) -> String {
    let visible_skills: Vec<&Skill> = skills
        .iter()
        .filter(|s| !s.disable_model_invocation)
        .collect();

    if visible_skills.is_empty() {
        return String::new();
    }

    let mut lines = vec![
        "\n\nThe following skills provide specialized instructions for specific tasks.".to_string(),
        if file_read_tool == "read" {
            "Use the read tool to load a skill's file when the task matches its description.".to_string()
        } else {
            "Use bash to load a skill's file when the task matches its description.".to_string()
        },
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_string(),
        String::new(),
        "<available_skills>".to_string(),
    ];

    for skill in visible_skills {
        lines.push("  <skill>".to_string());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path)
        ));
        lines.push("  </skill>".to_string());
    }

    lines.push("</available_skills>".to_string());

    lines.join("\n")
}

/// Escape the five XML entities, upstream's `escapeXml`.
#[must_use]
fn escape_xml(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// The load options, upstream's `LoadSkillsOptions`.
#[derive(Debug)]
pub struct LoadSkillsOptions<'a> {
    /// The working directory for project-local skills.
    pub cwd: &'a str,
    /// The agent config directory for global skills.
    pub agent_dir: &'a str,
    /// The explicit skill paths (files or directories).
    pub skill_paths: &'a [String],
    /// Whether the default skills directories load.
    pub include_defaults: bool,
}

/// Load skills from all configured locations, upstream's `loadSkills`.
///
/// The default directories (the agent dir's `skills`, the project `.pi`
/// skills) load first when asked, then the explicit paths. The first
/// skill with a name keeps it — later same-name skills record a collision
/// diagnostic — and a file already loaded through one path skips when a
/// second path reaches it through a symlink.
#[must_use]
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's md ends-with probe is case-sensitive; the 1:1 shape keeps it"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the 1:1 restatement of upstream's loadSkills reads as one arm per source kind"
)]
pub fn load_skills(options: &LoadSkillsOptions<'_>) -> LoadSkillsResult {
    let home = crate::config::home_dir();
    let resolved_cwd = resolve_path(options.cwd, &crate::config::process_cwd(), &home);
    let resolved_agent_dir = resolve_path(options.agent_dir, &crate::config::process_cwd(), &home);

    let mut skill_map: IndexMap<String, Skill> = IndexMap::new();
    let mut real_path_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut all_diagnostics: Vec<ResourceDiagnostic> = Vec::new();
    let mut collision_diagnostics: Vec<ResourceDiagnostic> = Vec::new();

    if options.include_defaults {
        let agent_skills_dir = Path::new(&resolved_agent_dir)
            .join("skills")
            .to_string_lossy()
            .into_owned();
        let mut matcher = IgnoreMatcher::new();
        let (skills, diagnostics) = load_skills_from_dir_internal(
            &agent_skills_dir,
            "user",
            true,
            &mut matcher,
            &agent_skills_dir,
        );
        add_skills(
            LoadSkillsResult {
                skills,
                diagnostics,
            },
            &mut skill_map,
            &mut real_path_set,
            &mut all_diagnostics,
            &mut collision_diagnostics,
        );
        let project_skills_dir = Path::new(&resolved_cwd)
            .join(crate::config::CONFIG_DIR_NAME)
            .join("skills")
            .to_string_lossy()
            .into_owned();
        let mut matcher = IgnoreMatcher::new();
        let (skills, diagnostics) = load_skills_from_dir_internal(
            &project_skills_dir,
            "project",
            true,
            &mut matcher,
            &project_skills_dir,
        );
        add_skills(
            LoadSkillsResult {
                skills,
                diagnostics,
            },
            &mut skill_map,
            &mut real_path_set,
            &mut all_diagnostics,
            &mut collision_diagnostics,
        );
    }

    let user_skills_dir = Path::new(&resolved_agent_dir)
        .join("skills")
        .to_string_lossy()
        .into_owned();
    let project_skills_dir = Path::new(&resolved_cwd)
        .join(crate::config::CONFIG_DIR_NAME)
        .join("skills")
        .to_string_lossy()
        .into_owned();

    let get_source = |resolved_path: &str| -> &'static str {
        if !options.include_defaults {
            if is_under_path(resolved_path, &user_skills_dir) {
                return "user";
            }
            if is_under_path(resolved_path, &project_skills_dir) {
                return "project";
            }
        }
        "path"
    };

    for raw_path in options.skill_paths {
        let resolved_path = resolve_path_with(
            raw_path,
            &resolved_cwd,
            &PathInputOptions {
                trim: true,
                ..PathInputOptions::default()
            },
        )
        .unwrap_or_else(|_| raw_path.clone());
        if !Path::new(&resolved_path).exists() {
            all_diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Warning,
                message: "skill path does not exist".to_string(),
                path: Some(resolved_path.clone()),
                collision: None,
            });
            continue;
        }

        let Ok(stats) = std::fs::metadata(&resolved_path) else {
            all_diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Warning,
                message: "failed to read skill path".to_string(),
                path: Some(resolved_path.clone()),
                collision: None,
            });
            continue;
        };

        let source = get_source(&resolved_path);
        if stats.is_dir() {
            let mut matcher = IgnoreMatcher::new();
            let (skills, diagnostics) = load_skills_from_dir_internal(
                &resolved_path,
                source,
                true,
                &mut matcher,
                &resolved_path,
            );
            add_skills(
                LoadSkillsResult {
                    skills,
                    diagnostics,
                },
                &mut skill_map,
                &mut real_path_set,
                &mut all_diagnostics,
                &mut collision_diagnostics,
            );
        } else if stats.is_file() && resolved_path.ends_with(".md") {
            let (skill, diagnostics) = load_skill_from_file(&resolved_path, source);
            if let Some(skill) = skill {
                add_skills(
                    LoadSkillsResult {
                        skills: vec![skill],
                        diagnostics,
                    },
                    &mut skill_map,
                    &mut real_path_set,
                    &mut all_diagnostics,
                    &mut collision_diagnostics,
                );
            } else {
                all_diagnostics.extend(diagnostics);
            }
        } else {
            all_diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Warning,
                message: "skill path is not a markdown file".to_string(),
                path: Some(resolved_path.clone()),
                collision: None,
            });
        }
    }

    LoadSkillsResult {
        skills: skill_map.into_values().collect(),
        diagnostics: all_diagnostics
            .into_iter()
            .chain(collision_diagnostics)
            .collect(),
    }
}

/// Fold one load's output into the accumulating maps, upstream's
/// `addSkills` closure: diagnostics flow through wholesale, symlinked
/// duplicates skip silently, and a same-name skill records a collision
/// while the first keeps the slot.
fn add_skills(
    result: LoadSkillsResult,
    skill_map: &mut IndexMap<String, Skill>,
    real_path_set: &mut std::collections::HashSet<String>,
    all_diagnostics: &mut Vec<ResourceDiagnostic>,
    collision_diagnostics: &mut Vec<ResourceDiagnostic>,
) {
    all_diagnostics.extend(result.diagnostics);
    for skill in result.skills {
        // Resolve symlinks to detect duplicate files.
        let real_path = canonicalize_path(&skill.file_path);

        // Skip silently if we've already loaded this exact file (via
        // symlink).
        if real_path_set.contains(&real_path) {
            continue;
        }

        if let Some(existing) = skill_map.get(&skill.name) {
            collision_diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Collision,
                message: format!("name \"{}\" collision", skill.name),
                path: Some(skill.file_path.clone()),
                collision: Some(ResourceCollision {
                    resource_type: ResourceType::Skill,
                    name: skill.name.clone(),
                    winner_path: existing.file_path.clone(),
                    loser_path: skill.file_path.clone(),
                    winner_source: None,
                    loser_source: None,
                }),
            });
        } else {
            skill_map.insert(skill.name.clone(), skill);
            real_path_set.insert(real_path);
        }
    }
}

//! The resource loader, upstream's `src/core/resource-loader.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! One reload discovers every resource layer: the skills, prompts, and
//! themes the package manager resolves (settings entries, the global
//! `~/.pi/agent/{skills,prompts,themes,extensions}` directories, the
//! project `.pi/{...}` tree gated on project trust), the context files
//! (`AGENTS.override.md`/`AGENTS.md`/`CLAUDE.md` up the directory chain,
//! with the linked-worktree dedup), and the `SYSTEM.md` /
//! `APPEND_SYSTEM.md` prompts, each stamped with its provenance.
//!
//! Two seams ride later tickets and stand in behind closures here:
//! extension loading (the out-of-process runtime and its cache, the
//! conflict diagnostics over its registrations, and the inline factories —
//! ticket #128), and the theme parser (the schema-validating loader; the
//! stand-in reads only the theme's name, what the loader dedupes on —
//! ticket #132). Upstream's `resetTimings` instrumentation and its timings
//! module have no counterpart: this port displays no number no provider or
//! session backend sends.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;

use crate::config::CONFIG_DIR_NAME;
use crate::diagnostics::{
    ResourceCollision, ResourceDiagnostic, ResourceDiagnosticKind, ResourceType,
};
use crate::footer_data_provider::{GitPaths, find_git_paths};
use crate::package_manager::{
    DefaultPackageManager, PackageManagerOptions, PathMetadata, ResolvedResource,
};
use crate::prompt_templates::{PromptTemplate, load_prompt_templates};
use crate::settings_manager::{FileSettingsStorage, SettingsManager, SettingsStorage};
use crate::skills::{Skill, load_skills};
use crate::source_info::{SourceInfo, SourceOrigin, SourceScope, create_source_info};
use crate::utils::paths::{
    PathInputOptions, basename_posix, canonicalize_path, is_local_path, is_under_path,
    resolve_path, resolve_path_with,
};
use crate::utils::text::strip_bom;

/// The boxed future the async seams return.
pub type BoxedFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// The boxed extension-load result.
pub type BoxedExtensionLoad = BoxedFuture<ExtensionLoadResult>;

/// The boxed trust decision.
pub type BoxedTrustDecision = BoxedFuture<bool>;

// =============================================================================
// Extension seam types, upstream's extensions/loader.ts surface (#128)
// =============================================================================

/// One loaded extension entry, upstream's `Extension` reduced to the
/// fields the loader layer reads and stamps.
///
/// Its declared path, the resolved path, the hidden flag, and the
/// provenance the loader applies. The registration maps (commands,
/// tools, flags) and the runtime state ride #128.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionEntry {
    /// The path the extension was requested by — a file path or an
    /// `<inline:N>` factory label.
    pub path: String,
    /// The resolved load path, upstream's `resolvedPath`.
    pub resolved_path: String,
    /// Whether the extension stays out of user-facing lists, upstream's
    /// `hidden` (inline factories only).
    pub hidden: bool,
    /// The provenance the loader stamped, upstream's `sourceInfo`.
    pub source_info: Option<SourceInfo>,
}

/// One extension load failure, upstream's `{ path, error }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionLoadError {
    /// The path that failed.
    pub path: String,
    /// The failure message.
    pub error: String,
}

/// The extension runtime handle, upstream's `ExtensionRuntime`.
///
/// The registrations and dispatch live with #128's loader; this
/// placeholder keeps the result shape final so that ticket swaps the seam
/// without reshaping the loader.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtensionRuntimeHandle;

/// The result of one extension load pass, upstream's
/// `LoadExtensionsResult`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtensionLoadResult {
    /// The extensions that loaded.
    pub extensions: Vec<ExtensionEntry>,
    /// The failures, conflicts included once #128 lands detection.
    pub errors: Vec<ExtensionLoadError>,
    /// The shared runtime the extensions registered into.
    pub runtime: ExtensionRuntimeHandle,
}

/// The extension-load seam, upstream's `loadExtensionsCached`. #128
/// replaces the default (an empty result) with the out-of-process loader
/// and its cache.
pub type LoadExtensionsFn = Arc<dyn Fn(Vec<String>, String) -> BoxedExtensionLoad + Send + Sync>;

/// The pre-trust decision callback, upstream's `resolveProjectTrust`
/// reload option: given the extensions loaded before project trust
/// resolved, decide whether the project is trusted.
pub type ResolveProjectTrustFn = Box<dyn Fn(ExtensionLoadResult) -> BoxedTrustDecision + Send>;

/// The reload options, upstream's `ResourceLoaderReloadOptions`.
#[derive(Default)]
pub struct ResourceLoaderReloadOptions {
    /// The trust decision callback, upstream's `resolveProjectTrust`.
    pub resolve_project_trust: Option<ResolveProjectTrustFn>,
}

impl std::fmt::Debug for ResourceLoaderReloadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceLoaderReloadOptions")
            .field(
                "resolve_project_trust",
                &self.resolve_project_trust.as_ref().map(|_| "<callback>"),
            )
            .finish()
    }
}

// =============================================================================
// Theme seam types (#132)
// =============================================================================

/// The theme-load seam, upstream's `loadThemeFromPath`.
///
/// Given a theme file's path, produce the theme's name — the piece the
/// loader dedupes on — or the failure message. #132 replaces the default
/// (a JSON `name` read) with the schema-validating parser and the full
/// theme.
pub type ThemeLoadFn = Arc<dyn Fn(&str) -> Result<Option<String>, String> + Send + Sync>;

/// One loaded theme, upstream's `Theme` reduced to the fields the loader
/// layer reads and stamps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThemeHandle {
    /// The theme's `name`; an unnamed theme dedupes as `unnamed`.
    pub name: Option<String>,
    /// The file the theme loaded from, upstream's `sourcePath`.
    pub source_path: Option<String>,
    /// The provenance the loader stamped, upstream's `sourceInfo`.
    pub source_info: Option<SourceInfo>,
}

/// The stand-in theme loader until #132: read the file, parse it as JSON,
/// take its `name` when one is a string. Upstream's full parser validates
/// the theme schema; this stand-in answers only what the loader layer
/// needs.
fn default_theme_load_fn() -> ThemeLoadFn {
    Arc::new(|theme_path: &str| {
        let content = std::fs::read_to_string(theme_path)
            .map_err(|error| format!("failed to read theme: {error}"))?;
        let parsed: serde_json::Value =
            serde_json::from_str(strip_bom(&content)).map_err(|error| error.to_string())?;
        Ok(parsed
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string))
    })
}

// =============================================================================
// Resource inputs
// =============================================================================

/// One extension-supplied resource path with its package metadata,
/// upstream's `{ path, metadata: PathMetadata }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathWithMetadata {
    /// The resource path (a file or directory).
    pub path: String,
    /// The package metadata the extension declared.
    pub metadata: PathMetadata,
}

/// The extension-supplied resource paths, upstream's
/// `ResourceExtensionPaths`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceExtensionPaths {
    /// Skills to add.
    pub skill_paths: Vec<PathWithMetadata>,
    /// Prompt templates to add.
    pub prompt_paths: Vec<PathWithMetadata>,
    /// Themes to add.
    pub theme_paths: Vec<PathWithMetadata>,
}

/// The loader options, upstream's `DefaultResourceLoaderOptions` minus the
/// settings manager the constructors take and the extension factories
/// that ride #128.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the option shape mirrors upstream's no* skip flags"
)]
pub struct DefaultResourceLoaderOptions {
    /// The working directory.
    pub cwd: String,
    /// The agent config directory.
    pub agent_dir: String,
    /// Additional extension paths from the CLI, upstream's
    /// `additionalExtensionPaths`.
    pub additional_extension_paths: Vec<String>,
    /// Additional skill paths from the CLI, upstream's
    /// `additionalSkillPaths`.
    pub additional_skill_paths: Vec<String>,
    /// Additional prompt template paths, upstream's
    /// `additionalPromptTemplatePaths`.
    pub additional_prompt_template_paths: Vec<String>,
    /// Additional theme paths, upstream's `additionalThemePaths`.
    pub additional_theme_paths: Vec<String>,
    /// Skip extension discovery, upstream's `noExtensions`.
    pub no_extensions: bool,
    /// Skip skill discovery, upstream's `noSkills`.
    pub no_skills: bool,
    /// Skip prompt template discovery, upstream's `noPromptTemplates`.
    pub no_prompt_templates: bool,
    /// Skip theme discovery, upstream's `noThemes`.
    pub no_themes: bool,
    /// Skip context file discovery, upstream's `noContextFiles`.
    pub no_context_files: bool,
    /// A system prompt source — a literal prompt or a file path, upstream's
    /// `systemPrompt`.
    pub system_prompt: Option<String>,
    /// Append system prompt sources, upstream's `appendSystemPrompt`.
    pub append_system_prompt: Option<Vec<String>>,
    /// The theme-load seam; defaults to the stand-in name reader (#132).
    pub theme_load_fn: Option<ThemeLoadFn>,
    /// The extension-load seam; defaults to an empty result (#128).
    pub load_extensions_fn: Option<LoadExtensionsFn>,
    /// The skills override, upstream's `skillsOverride`.
    pub skills_override: Option<SkillsOverrideFn>,
    /// The prompts override, upstream's `promptsOverride`.
    pub prompts_override: Option<PromptsOverrideFn>,
    /// The themes override, upstream's `themesOverride`.
    pub themes_override: Option<ThemesOverrideFn>,
    /// The context-files override, upstream's `agentsFilesOverride`.
    pub agents_files_override: Option<AgentsFilesOverrideFn>,
    /// The system prompt override, upstream's `systemPromptOverride`.
    pub system_prompt_override: Option<SystemPromptOverrideFn>,
    /// The append system prompt override, upstream's
    /// `appendSystemPromptOverride`.
    pub append_system_prompt_override: Option<AppendSystemPromptOverrideFn>,
}

impl std::fmt::Debug for DefaultResourceLoaderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultResourceLoaderOptions")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .field(
                "additional_extension_paths",
                &self.additional_extension_paths,
            )
            .field("additional_skill_paths", &self.additional_skill_paths)
            .field(
                "additional_prompt_template_paths",
                &self.additional_prompt_template_paths,
            )
            .field("additional_theme_paths", &self.additional_theme_paths)
            .field("no_extensions", &self.no_extensions)
            .field("no_skills", &self.no_skills)
            .field("no_prompt_templates", &self.no_prompt_templates)
            .field("no_themes", &self.no_themes)
            .field("no_context_files", &self.no_context_files)
            .field("system_prompt", &self.system_prompt)
            .field("append_system_prompt", &self.append_system_prompt)
            .field(
                "theme_load_fn",
                &self.theme_load_fn.as_ref().map(|_| "<seam>"),
            )
            .field(
                "load_extensions_fn",
                &self.load_extensions_fn.as_ref().map(|_| "<seam>"),
            )
            .field(
                "skills_override",
                &self.skills_override.as_ref().map(|_| "<override>"),
            )
            .field(
                "prompts_override",
                &self.prompts_override.as_ref().map(|_| "<override>"),
            )
            .field(
                "themes_override",
                &self.themes_override.as_ref().map(|_| "<override>"),
            )
            .field(
                "agents_files_override",
                &self.agents_files_override.as_ref().map(|_| "<override>"),
            )
            .field(
                "system_prompt_override",
                &self.system_prompt_override.as_ref().map(|_| "<override>"),
            )
            .field(
                "append_system_prompt_override",
                &self
                    .append_system_prompt_override
                    .as_ref()
                    .map(|_| "<override>"),
            )
            .finish()
    }
}

/// The skills override, upstream's `skillsOverride`.
pub type SkillsOverrideFn = Box<dyn Fn(LoadedSkillsResult) -> LoadedSkillsResult + Send + Sync>;

/// The prompts override, upstream's `promptsOverride`.
pub type PromptsOverrideFn = Box<dyn Fn(LoadedPromptsResult) -> LoadedPromptsResult + Send + Sync>;

/// The themes override, upstream's `themesOverride`.
pub type ThemesOverrideFn = Box<dyn Fn(LoadedThemesResult) -> LoadedThemesResult + Send + Sync>;

/// The context-files override, upstream's `agentsFilesOverride`.
pub type AgentsFilesOverrideFn = Box<dyn Fn(Vec<ContextFile>) -> Vec<ContextFile> + Send + Sync>;

/// The system prompt override, upstream's `systemPromptOverride`.
pub type SystemPromptOverrideFn = Box<dyn Fn(Option<String>) -> Option<String> + Send + Sync>;

/// The append system prompt override, upstream's
/// `appendSystemPromptOverride`.
pub type AppendSystemPromptOverrideFn = Box<dyn Fn(Vec<String>) -> Vec<String> + Send + Sync>;

/// The skills-plus-diagnostics result the overrides reshape, upstream's
/// `{ skills, diagnostics }`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadedSkillsResult {
    /// The skills.
    pub skills: Vec<Skill>,
    /// The diagnostics.
    pub diagnostics: Vec<ResourceDiagnostic>,
}

/// The prompts-plus-diagnostics result the overrides reshape.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadedPromptsResult {
    /// The prompts.
    pub prompts: Vec<PromptTemplate>,
    /// The diagnostics.
    pub diagnostics: Vec<ResourceDiagnostic>,
}

/// The themes-plus-diagnostics result the overrides reshape.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadedThemesResult {
    /// The themes.
    pub themes: Vec<ThemeHandle>,
    /// The diagnostics.
    pub diagnostics: Vec<ResourceDiagnostic>,
}

/// A project context file, upstream's `{ path, content }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextFile {
    /// The absolute path the file loaded from.
    pub path: String,
    /// The file's text, byte-order mark stripped.
    pub content: String,
}

/// The discovered system prompt source, upstream's
/// `{ path: string } | undefined`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptSource {
    /// The file the prompt came from.
    pub path: String,
}

// =============================================================================
// Standalone loaders, upstream's module-level functions
// =============================================================================

/// Resolve a prompt input as a file path or literal text, upstream's
/// `resolvePromptInput`: an existing path reads (BOM-stripped), a read
/// failure warns and falls back to the literal, anything else stays
/// literal.
fn resolve_prompt_input(input: Option<&str>, description: &str) -> Option<String> {
    let input = input?;
    if Path::new(input).exists() {
        match std::fs::read_to_string(input) {
            Ok(content) => return Some(strip_bom(&content).to_string()),
            Err(error) => {
                #[expect(
                    clippy::print_stderr,
                    reason = "upstream warns through console.error before falling back to the literal"
                )]
                {
                    eprintln!("Warning: Could not read {description} file {input}: {error}");
                }
                return Some(input.to_string());
            }
        }
    }

    Some(input.to_string())
}

/// Load a directory's context file, upstream's `loadContextFileFromDir`:
/// the first existing candidate that stats as a file wins. A stat or read
/// failure warns and the walk moves to the next candidate.
fn load_context_file_from_dir(dir: &str) -> Option<ContextFile> {
    let candidates = [
        "AGENTS.override.md",
        "AGENTS.md",
        "AGENTS.MD",
        "CLAUDE.md",
        "CLAUDE.MD",
    ];
    for filename in candidates {
        let file_path = Path::new(dir).join(filename);
        if !file_path.exists() {
            continue;
        }
        match std::fs::metadata(&file_path).map(|m| m.is_file()) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(error) => {
                #[expect(
                    clippy::print_stderr,
                    reason = "upstream warns through console.error and moves to the next candidate"
                )]
                {
                    eprintln!(
                        "Warning: Could not read {}: {error}",
                        file_path.to_string_lossy()
                    );
                }
                continue;
            }
        }
        match std::fs::read_to_string(&file_path) {
            Ok(content) => {
                return Some(ContextFile {
                    path: file_path.to_string_lossy().into_owned(),
                    content: strip_bom(&content).to_string(),
                });
            }
            Err(error) => {
                #[expect(
                    clippy::print_stderr,
                    reason = "upstream warns through console.error and moves to the next candidate"
                )]
                {
                    eprintln!(
                        "Warning: Could not read {}: {error}",
                        file_path.to_string_lossy()
                    );
                }
            }
        }
    }
    None
}

/// The main repo's context file that a nested linked worktree's own copy
/// shadows, upstream's `findShadowedContextFile`: both occupy the same
/// logical repository scope, so loading both applies that context twice.
/// `None` when nothing is shadowed, leaving normal ancestor inheritance
/// alone.
///
/// The shadow target is canonicalized (realpath) because `git worktree
/// add` writes the `.git` file's `gitdir:` target in realpath form while
/// cwd may still be symlinked (macOS `/tmp` -> `/private/tmp`).
fn find_shadowed_context_file(cwd: &str) -> Option<String> {
    let git_paths: GitPaths = find_git_paths(cwd)?;
    let common_git_dir = canonicalize_path(&git_paths.common_git_dir);
    let worktree_root = canonicalize_path(&git_paths.repo_dir);
    let main_repo_root = dirname_posix_loader(&common_git_dir);
    // False for an ordinary repo, where the two are the same dir, and for
    // a sibling worktree (`git worktree add ../feat`), whose main repo is
    // not an ancestor.
    if !worktree_root.starts_with(&format!("{main_repo_root}/")) {
        return None;
    }
    // dirname of the common git dir is the main worktree root only when
    // that dir is itself checked out from the same repo. In a bare layout
    // (`proj/.bare` + `proj/main`) it is just the directory holding
    // `.bare`, which tracks nothing; a submodule's gitdir has no
    // `commondir`, so it lands under `.git/modules`.
    let main_repo_git = canonicalize_path(&format!("{main_repo_root}/.git"));
    if main_repo_git != common_git_dir {
        return None;
    }
    load_context_file_from_dir(&worktree_root).map(|worktree_context_file| {
        format!(
            "{main_repo_root}/{}",
            basename_posix(&worktree_context_file.path)
        )
    })
}

/// Load the project context files, upstream's `loadProjectContextFiles`.
///
/// The agent dir's context file first, then the cwd's ancestors
/// innermost-first with each directory's preferred candidate, skipping
/// the shadowed main-repo duplicate and files already seen by path.
#[must_use]
pub fn load_project_context_files(cwd: &str, agent_dir: &str) -> Vec<ContextFile> {
    let home = crate::config::home_dir();
    let base = crate::config::process_cwd();
    let resolved_cwd = resolve_path(cwd, &base, &home);
    let resolved_agent_dir = resolve_path(agent_dir, &base, &home);

    let mut context_files: Vec<ContextFile> = Vec::new();
    let mut seen_paths: HashSet<String> = HashSet::new();

    if let Some(global_context) = load_context_file_from_dir(&resolved_agent_dir) {
        seen_paths.insert(global_context.path.clone());
        context_files.push(global_context);
    }

    let mut ancestor_context_files: Vec<ContextFile> = Vec::new();

    let shadowed_context_file = find_shadowed_context_file(&resolved_cwd);
    let mut current_dir = resolved_cwd;

    loop {
        let context_file = load_context_file_from_dir(&current_dir);
        let is_shadowed = shadowed_context_file.as_ref().is_some_and(|shadowed| {
            canonicalize_path(
                context_file
                    .as_ref()
                    .map(|file| file.path.as_str())
                    .unwrap_or_default(),
            ) == *shadowed
        });
        if let Some(context_file) = context_file
            && !is_shadowed
            && !seen_paths.contains(&context_file.path)
        {
            seen_paths.insert(context_file.path.clone());
            ancestor_context_files.insert(0, context_file);
        }

        let Some(parent_dir) = parent_dir_posix(&current_dir) else {
            break;
        };
        current_dir = parent_dir;
    }

    context_files.extend(ancestor_context_files);

    context_files
}

// =============================================================================
// DefaultResourceLoader
// =============================================================================

/// The resource loader, upstream's `DefaultResourceLoader`. Results start
/// empty and fill on [`DefaultResourceLoader::reload`].
#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields mirror upstream's no* skip flags on the loader class"
)]
pub struct DefaultResourceLoader<S: SettingsStorage> {
    cwd: String,
    agent_dir: String,
    // The manager and the loader share one settings manager, upstream's
    // constructor passing the same instance to both.
    settings_manager: Arc<Mutex<SettingsManager<S>>>,
    package_manager: DefaultPackageManager<S>,
    additional_extension_paths: Vec<String>,
    additional_skill_paths: Vec<String>,
    additional_prompt_template_paths: Vec<String>,
    additional_theme_paths: Vec<String>,
    no_extensions: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    no_themes: bool,
    no_context_files: bool,
    system_prompt_source: Option<String>,
    append_system_prompt_source: Option<Vec<String>>,
    skills_override: Option<SkillsOverrideFn>,
    prompts_override: Option<PromptsOverrideFn>,
    themes_override: Option<ThemesOverrideFn>,
    agents_files_override: Option<AgentsFilesOverrideFn>,
    system_prompt_override: Option<SystemPromptOverrideFn>,
    append_system_prompt_override: Option<AppendSystemPromptOverrideFn>,
    load_extensions_fn: LoadExtensionsFn,
    theme_load_fn: ThemeLoadFn,

    extensions_result: ExtensionLoadResult,
    skills: Vec<Skill>,
    skill_diagnostics: Vec<ResourceDiagnostic>,
    prompts: Vec<PromptTemplate>,
    prompt_diagnostics: Vec<ResourceDiagnostic>,
    themes: Vec<ThemeHandle>,
    theme_diagnostics: Vec<ResourceDiagnostic>,
    agents_files: Vec<ContextFile>,
    system_prompt: Option<String>,
    system_prompt_source_path: Option<String>,
    append_system_prompt: Vec<String>,
    append_system_prompt_source_paths: Vec<String>,
    last_skill_paths: Vec<String>,
    extension_skill_source_infos: HashMap<String, SourceInfo>,
    extension_prompt_source_infos: HashMap<String, SourceInfo>,
    extension_theme_source_infos: HashMap<String, SourceInfo>,
    resource_metadata_by_path: HashMap<String, PathMetadata>,
    last_prompt_paths: Vec<String>,
    last_theme_paths: Vec<String>,
}

impl<S: SettingsStorage + 'static> DefaultResourceLoader<S> {
    /// The loader over an injected settings manager, the seam the in-memory
    /// tests and non-file storages ride.
    #[must_use]
    pub fn new(
        options: DefaultResourceLoaderOptions,
        settings_manager: SettingsManager<S>,
    ) -> Self {
        let cwd = resolve_default(&options.cwd);
        let agent_dir = resolve_default(&options.agent_dir);
        let settings_manager = Arc::new(Mutex::new(settings_manager));
        let package_manager = DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            settings: Arc::clone(&settings_manager),
            command_runner: None,
            env: None,
            http_client: None,
        });
        Self {
            additional_extension_paths: options.additional_extension_paths,
            additional_skill_paths: options.additional_skill_paths,
            additional_prompt_template_paths: options.additional_prompt_template_paths,
            additional_theme_paths: options.additional_theme_paths,
            no_extensions: options.no_extensions,
            no_skills: options.no_skills,
            no_prompt_templates: options.no_prompt_templates,
            no_themes: options.no_themes,
            no_context_files: options.no_context_files,
            system_prompt_source: options.system_prompt,
            append_system_prompt_source: options.append_system_prompt,
            skills_override: options.skills_override,
            prompts_override: options.prompts_override,
            themes_override: options.themes_override,
            agents_files_override: options.agents_files_override,
            system_prompt_override: options.system_prompt_override,
            append_system_prompt_override: options.append_system_prompt_override,
            load_extensions_fn: options
                .load_extensions_fn
                .unwrap_or_else(default_load_extensions_fn),
            theme_load_fn: options.theme_load_fn.unwrap_or_else(default_theme_load_fn),
            settings_manager,
            package_manager,
            cwd,
            agent_dir,
            extensions_result: ExtensionLoadResult::default(),
            skills: Vec::new(),
            skill_diagnostics: Vec::new(),
            prompts: Vec::new(),
            prompt_diagnostics: Vec::new(),
            themes: Vec::new(),
            theme_diagnostics: Vec::new(),
            agents_files: Vec::new(),
            system_prompt: None,
            system_prompt_source_path: None,
            append_system_prompt: Vec::new(),
            append_system_prompt_source_paths: Vec::new(),
            last_skill_paths: Vec::new(),
            extension_skill_source_infos: HashMap::new(),
            extension_prompt_source_infos: HashMap::new(),
            extension_theme_source_infos: HashMap::new(),
            resource_metadata_by_path: HashMap::new(),
            last_prompt_paths: Vec::new(),
            last_theme_paths: Vec::new(),
        }
    }

    /// The loaded extensions, upstream's `getExtensions`.
    #[must_use]
    pub fn get_extensions(&self) -> ExtensionLoadResult {
        self.extensions_result.clone()
    }

    /// The loaded skills, upstream's `getSkills`.
    #[must_use]
    pub fn get_skills(&self) -> LoadedSkillsResult {
        LoadedSkillsResult {
            skills: self.skills.clone(),
            diagnostics: self.skill_diagnostics.clone(),
        }
    }

    /// The loaded prompt templates, upstream's `getPrompts`.
    #[must_use]
    pub fn get_prompts(&self) -> LoadedPromptsResult {
        LoadedPromptsResult {
            prompts: self.prompts.clone(),
            diagnostics: self.prompt_diagnostics.clone(),
        }
    }

    /// The loaded themes, upstream's `getThemes`.
    #[must_use]
    pub fn get_themes(&self) -> LoadedThemesResult {
        LoadedThemesResult {
            themes: self.themes.clone(),
            diagnostics: self.theme_diagnostics.clone(),
        }
    }

    /// The loaded context files, upstream's `getAgentsFiles`.
    #[must_use]
    pub fn get_agents_files(&self) -> Vec<ContextFile> {
        self.agents_files.clone()
    }

    /// The resolved system prompt, upstream's `getSystemPrompt`.
    #[must_use]
    pub fn get_system_prompt(&self) -> Option<String> {
        self.system_prompt.clone()
    }

    /// The file the system prompt came from, upstream's
    /// `getSystemPromptSource`; literal prompts carry no source.
    #[must_use]
    pub fn get_system_prompt_source(&self) -> Option<PromptSource> {
        self.system_prompt_source_path
            .as_ref()
            .map(|path| PromptSource { path: path.clone() })
    }

    /// The append system prompts, upstream's `getAppendSystemPrompt`.
    #[must_use]
    pub fn get_append_system_prompt(&self) -> Vec<String> {
        self.append_system_prompt.clone()
    }

    /// The files the append prompts came from, upstream's
    /// `getAppendSystemPromptSources`.
    #[must_use]
    pub fn get_append_system_prompt_sources(&self) -> Vec<PromptSource> {
        self.append_system_prompt_source_paths
            .iter()
            .map(|path| PromptSource { path: path.clone() })
            .collect()
    }

    /// Add extension-supplied resources, upstream's `extendResources`:
    /// record their provenance, merge the paths into the running sets,
    /// and re-run the affected loads. Paths normalize against the cwd,
    /// base dirs included.
    pub fn extend_resources(&mut self, paths: &ResourceExtensionPaths) {
        let skill_paths = self.normalize_extension_paths(&paths.skill_paths);
        let prompt_paths = self.normalize_extension_paths(&paths.prompt_paths);
        let theme_paths = self.normalize_extension_paths(&paths.theme_paths);

        for entry in &skill_paths {
            self.extension_skill_source_infos.insert(
                entry.path.clone(),
                create_source_info(&entry.path, &entry.metadata),
            );
        }
        for entry in &prompt_paths {
            self.extension_prompt_source_infos.insert(
                entry.path.clone(),
                create_source_info(&entry.path, &entry.metadata),
            );
        }
        for entry in &theme_paths {
            self.extension_theme_source_infos.insert(
                entry.path.clone(),
                create_source_info(&entry.path, &entry.metadata),
            );
        }

        if !skill_paths.is_empty() {
            self.last_skill_paths = self.merge_paths(
                &self.last_skill_paths.clone(),
                &skill_paths
                    .iter()
                    .map(|entry| entry.path.clone())
                    .collect::<Vec<_>>(),
            );
            let metadata = self.resource_metadata_by_path.clone();
            self.update_skills_from_paths(&self.last_skill_paths.clone(), Some(&metadata));
        }

        if !prompt_paths.is_empty() {
            self.last_prompt_paths = self.merge_paths(
                &self.last_prompt_paths.clone(),
                &prompt_paths
                    .iter()
                    .map(|entry| entry.path.clone())
                    .collect::<Vec<_>>(),
            );
            let metadata = self.resource_metadata_by_path.clone();
            self.update_prompts_from_paths(&self.last_prompt_paths.clone(), Some(&metadata));
        }

        if !theme_paths.is_empty() {
            self.last_theme_paths = self.merge_paths(
                &self.last_theme_paths.clone(),
                &theme_paths
                    .iter()
                    .map(|entry| entry.path.clone())
                    .collect::<Vec<_>>(),
            );
            let metadata = self.resource_metadata_by_path.clone();
            self.update_themes_from_paths(&self.last_theme_paths.clone(), Some(&metadata));
        }
    }

    /// Load the extension set with project settings forced untrusted,
    /// upstream's `loadProjectTrustExtensions`: the bootstrap pass keeps
    /// project-local extensions and packages out while still loading
    /// user/global and temporary CLI extensions.
    pub async fn load_project_trust_extensions(&mut self) -> ExtensionLoadResult {
        {
            let mut manager = self
                .settings_manager
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            manager.set_project_trusted(false);
            manager.reload();
        }
        self.load_current_extension_set().await
    }

    /// Reload every resource layer, upstream's `reload`.
    ///
    /// With a `resolve_project_trust` callback, the pre-trust bootstrap
    /// loads the untrusted extension set first, the callback decides, and
    /// the trusted reload follows. Project trust otherwise rides the
    /// settings manager's current state.
    ///
    /// # Panics
    /// A package-manager resolution failure panics with the manager's
    /// message: upstream's loader awaits `resolve()` and lets the
    /// rejection propagate, and this reload keeps its infallible
    /// signature — the session integration picks the catch policy.
    #[expect(
        clippy::too_many_lines,
        reason = "the 1:1 restatement of upstream's reload reads as one block per resource family"
    )]
    #[expect(
        clippy::expect_used,
        reason = "the reload is infallible by signature; see # Panics for the trade-off"
    )]
    pub async fn reload(&mut self, options: Option<ResourceLoaderReloadOptions>) {
        // Upstream clears the extension module cache between reloads; the
        // cache lives with #128's loader, which owns its invalidation.

        let resolve_project_trust = options
            .as_ref()
            .and_then(|options| options.resolve_project_trust.as_ref());
        let pre_trust_extensions = if let Some(resolve_project_trust) = resolve_project_trust {
            let pre_trust = self.load_project_trust_extensions().await;
            let project_trusted = resolve_project_trust(pre_trust.clone()).await;
            self.settings_manager
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .set_project_trusted(project_trusted);
            Some(pre_trust)
        } else {
            None
        };

        // reload() preserves SettingsManager.projectTrusted and reloads
        // settings for that trust state.
        self.settings_manager
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .reload();
        let resolved_paths = self
            .package_manager
            .resolve(None)
            .await
            .expect("package-manager resolve failed");
        let cli_extension_paths = self
            .package_manager
            .resolve_extension_sources(&self.additional_extension_paths, false, true)
            .await
            .expect("package-manager resolve_extension_sources failed");
        // Kept on the instance so post-reload passes (extendResources) can
        // still resolve package metadata.
        self.resource_metadata_by_path = HashMap::new();

        self.extension_skill_source_infos = HashMap::new();
        self.extension_prompt_source_infos = HashMap::new();
        self.extension_theme_source_infos = HashMap::new();

        let mut metadata_by_path = self.resource_metadata_by_path.clone();

        // Extract the enabled paths, recording every path's metadata.
        let mut enabled_extension_resources: Vec<ResolvedResource> = Vec::new();
        for resource in &resolved_paths.extensions {
            metadata_by_path
                .entry(resource.path.clone())
                .or_insert_with(|| resource.metadata.clone());
            if resource.enabled {
                enabled_extension_resources.push(resource.clone());
            }
        }
        let enabled_extensions: Vec<String> = enabled_extension_resources
            .iter()
            .map(|resource| resource.path.clone())
            .collect();

        let mut enabled_skill_resources: Vec<ResolvedResource> = Vec::new();
        for resource in &resolved_paths.skills {
            metadata_by_path
                .entry(resource.path.clone())
                .or_insert_with(|| resource.metadata.clone());
            if resource.enabled {
                enabled_skill_resources.push(resource.clone());
            }
        }

        let mut enabled_prompt_paths: Vec<String> = Vec::new();
        for resource in &resolved_paths.prompts {
            metadata_by_path
                .entry(resource.path.clone())
                .or_insert_with(|| resource.metadata.clone());
            if resource.enabled {
                enabled_prompt_paths.push(resource.path.clone());
            }
        }

        let mut enabled_theme_paths: Vec<String> = Vec::new();
        for resource in &resolved_paths.themes {
            metadata_by_path
                .entry(resource.path.clone())
                .or_insert_with(|| resource.metadata.clone());
            if resource.enabled {
                enabled_theme_paths.push(resource.path.clone());
            }
        }

        let enabled_skills: Vec<String> = enabled_skill_resources
            .iter()
            .map(|resource| Self::map_skill_path(resource, &mut metadata_by_path))
            .collect();

        // Add CLI paths metadata.
        for resource in &cli_extension_paths.extensions {
            metadata_by_path
                .entry(resource.path.clone())
                .or_insert_with(|| PathMetadata {
                    source: "cli".to_string(),
                    scope: SourceScope::Temporary,
                    origin: SourceOrigin::TopLevel,
                    base_dir: None,
                });
        }
        for resource in &cli_extension_paths.skills {
            metadata_by_path
                .entry(resource.path.clone())
                .or_insert_with(|| PathMetadata {
                    source: "cli".to_string(),
                    scope: SourceScope::Temporary,
                    origin: SourceOrigin::TopLevel,
                    base_dir: None,
                });
        }

        let cli_enabled_extensions: Vec<String> = cli_extension_paths
            .extensions
            .iter()
            .filter(|resource| resource.enabled)
            .map(|resource| resource.path.clone())
            .collect();
        let cli_enabled_skills: Vec<String> = cli_extension_paths
            .skills
            .iter()
            .filter(|resource| resource.enabled)
            .map(|resource| resource.path.clone())
            .collect();
        let cli_enabled_prompts: Vec<String> = cli_extension_paths
            .prompts
            .iter()
            .filter(|resource| resource.enabled)
            .map(|resource| resource.path.clone())
            .collect();
        let cli_enabled_themes: Vec<String> = cli_extension_paths
            .themes
            .iter()
            .filter(|resource| resource.enabled)
            .map(|resource| resource.path.clone())
            .collect();

        let extension_paths = if self.no_extensions {
            cli_enabled_extensions
        } else {
            self.merge_paths(&cli_enabled_extensions, &enabled_extensions)
        };

        let mut extensions_result = self
            .load_final_extension_set(extension_paths, pre_trust_extensions.as_ref())
            .await;
        for p in &self.additional_extension_paths {
            if is_local_path(p) {
                let resolved = self.resolve_resource_path(p);
                if !Path::new(&resolved).exists() {
                    extensions_result.errors.push(ExtensionLoadError {
                        path: resolved.clone(),
                        error: format!("Extension path does not exist: {resolved}"),
                    });
                }
            }
        }
        for entry in &mut extensions_result.extensions {
            entry.source_info = Some(
                self.find_source_info_for_path(&entry.path, None, Some(&metadata_by_path))
                    .unwrap_or_else(|| self.get_default_source_info_for_path(&entry.path)),
            );
        }
        self.extensions_result = extensions_result;

        let skill_paths = if self.no_skills {
            self.merge_paths(&cli_enabled_skills, &self.additional_skill_paths)
        } else {
            let mut discovered = cli_enabled_skills.clone();
            discovered.extend(enabled_skills.clone());
            self.merge_paths(&discovered, &self.additional_skill_paths)
        };

        self.last_skill_paths.clone_from(&skill_paths);
        self.update_skills_from_paths(&skill_paths, Some(&metadata_by_path));
        for p in &self.additional_skill_paths {
            if is_local_path(p) {
                let resolved = self.resolve_resource_path(p);
                if !Path::new(&resolved).exists()
                    && !self
                        .skill_diagnostics
                        .iter()
                        .any(|d| d.path.as_deref() == Some(resolved.as_str()))
                {
                    self.skill_diagnostics.push(ResourceDiagnostic {
                        kind: ResourceDiagnosticKind::Error,
                        message: "Skill path does not exist".to_string(),
                        path: Some(resolved.clone()),
                        collision: None,
                    });
                }
            }
        }

        let prompt_paths = if self.no_prompt_templates {
            self.merge_paths(&cli_enabled_prompts, &self.additional_prompt_template_paths)
        } else {
            let mut discovered = cli_enabled_prompts.clone();
            discovered.extend(enabled_prompt_paths.clone());
            self.merge_paths(&discovered, &self.additional_prompt_template_paths)
        };

        self.last_prompt_paths.clone_from(&prompt_paths);
        self.update_prompts_from_paths(&prompt_paths, Some(&metadata_by_path));
        for p in &self.additional_prompt_template_paths {
            if is_local_path(p) {
                let resolved = self.resolve_resource_path(p);
                if !Path::new(&resolved).exists()
                    && !self
                        .prompt_diagnostics
                        .iter()
                        .any(|d| d.path.as_deref() == Some(resolved.as_str()))
                {
                    self.prompt_diagnostics.push(ResourceDiagnostic {
                        kind: ResourceDiagnosticKind::Error,
                        message: "Prompt template path does not exist".to_string(),
                        path: Some(resolved.clone()),
                        collision: None,
                    });
                }
            }
        }

        let theme_paths = if self.no_themes {
            self.merge_paths(&cli_enabled_themes, &self.additional_theme_paths)
        } else {
            let mut discovered = cli_enabled_themes.clone();
            discovered.extend(enabled_theme_paths.clone());
            self.merge_paths(&discovered, &self.additional_theme_paths)
        };

        self.last_theme_paths.clone_from(&theme_paths);
        self.update_themes_from_paths(&theme_paths, Some(&metadata_by_path));
        for p in &self.additional_theme_paths {
            let resolved = self.resolve_resource_path(p);
            if !Path::new(&resolved).exists()
                && !self
                    .theme_diagnostics
                    .iter()
                    .any(|d| d.path.as_deref() == Some(resolved.as_str()))
            {
                self.theme_diagnostics.push(ResourceDiagnostic {
                    kind: ResourceDiagnosticKind::Error,
                    message: "Theme path does not exist".to_string(),
                    path: Some(resolved.clone()),
                    collision: None,
                });
            }
        }

        let agents_files = if self.no_context_files {
            Vec::new()
        } else {
            load_project_context_files(&self.cwd, &self.agent_dir)
        };
        let resolved_agents_files = match &self.agents_files_override {
            Some(agents_files_override) => agents_files_override(agents_files),
            None => agents_files,
        };
        self.agents_files = resolved_agents_files;

        let system_prompt_source = self
            .system_prompt_source
            .clone()
            .or_else(|| self.discover_system_prompt_file());
        let base_system_prompt =
            resolve_prompt_input(system_prompt_source.as_deref(), "system prompt");
        self.system_prompt = match &self.system_prompt_override {
            Some(system_prompt_override) => system_prompt_override(base_system_prompt),
            None => base_system_prompt,
        };
        self.system_prompt_source_path = system_prompt_source
            .filter(|source| Path::new(source).exists())
            .map(|source| resolve_default(&source));

        let append_sources = self.append_system_prompt_source.clone().unwrap_or_else(|| {
            self.discover_append_system_prompt_file()
                .map(|discovered| vec![discovered])
                .unwrap_or_default()
        });
        let base_append: Vec<String> = append_sources
            .iter()
            .filter_map(|source| resolve_prompt_input(Some(source), "append system prompt"))
            .collect();
        self.append_system_prompt = match &self.append_system_prompt_override {
            Some(append_system_prompt_override) => append_system_prompt_override(base_append),
            None => base_append,
        };
        self.append_system_prompt_source_paths = append_sources
            .iter()
            .filter(|source| Path::new(source).exists())
            .map(|source| resolve_default(source))
            .collect();
    }

    /// The extension set for the pre-trust bootstrap, upstream's
    /// `loadCurrentExtensionSet` with the inline-factory arm riding #128.
    ///
    /// # Panics
    /// A package-manager resolution failure panics, the same
    /// infallible-signature restatement as [`Self::reload`].
    #[expect(
        clippy::expect_used,
        reason = "the bootstrap pass is reload's own failure surface; see Self::reload's # Panics"
    )]
    async fn load_current_extension_set(&self) -> ExtensionLoadResult {
        let resolved_paths = self
            .package_manager
            .resolve(None)
            .await
            .expect("package-manager resolve failed");
        let cli_extension_paths = self
            .package_manager
            .resolve_extension_sources(&self.additional_extension_paths, false, true)
            .await
            .expect("package-manager resolve_extension_sources failed");
        let enabled_extensions: Vec<String> = resolved_paths
            .extensions
            .iter()
            .filter(|resource| resource.enabled)
            .map(|resource| resource.path.clone())
            .collect();
        let cli_enabled_extensions: Vec<String> = cli_extension_paths
            .extensions
            .iter()
            .filter(|resource| resource.enabled)
            .map(|resource| resource.path.clone())
            .collect();
        let extension_paths = if self.no_extensions {
            cli_enabled_extensions
        } else {
            self.merge_paths(&cli_enabled_extensions, &enabled_extensions)
        };
        self.call_load_extensions(&extension_paths).await
    }

    /// The final extension set, upstream's `loadFinalExtensionSet`: with a
    /// pre-trust result, the extensions it already loaded (and the paths
    /// it already failed) skip the second load and re-order into the
    /// merged path order.
    async fn load_final_extension_set(
        &self,
        extension_paths: Vec<String>,
        pre_trust_extensions: Option<&ExtensionLoadResult>,
    ) -> ExtensionLoadResult {
        let Some(pre_trust) = pre_trust_extensions else {
            return self.call_load_extensions(&extension_paths).await;
        };

        let mut preloaded_by_path: HashMap<String, ExtensionEntry> = pre_trust
            .extensions
            .iter()
            .filter(|extension| !extension.path.starts_with("<inline:"))
            .map(|extension| (extension.resolved_path.clone(), extension.clone()))
            .collect();
        let failed_preload_paths: HashSet<String> = pre_trust
            .errors
            .iter()
            .map(|error| self.resolve_extension_load_path(&error.path))
            .collect();
        let remaining_paths: Vec<String> = extension_paths
            .iter()
            .filter(|path| {
                let resolved_path = self.resolve_extension_load_path(path);
                !preloaded_by_path.contains_key(&resolved_path)
                    && !failed_preload_paths.contains(&resolved_path)
            })
            .cloned()
            .collect();
        let remaining_extensions = self.call_load_extensions(&remaining_paths).await;
        for extension in &remaining_extensions.extensions {
            preloaded_by_path.insert(extension.resolved_path.clone(), extension.clone());
        }

        let inline_extensions: Vec<ExtensionEntry> = pre_trust
            .extensions
            .iter()
            .filter(|extension| extension.path.starts_with("<inline:"))
            .cloned()
            .collect();
        let mut ordered_extensions: Vec<ExtensionEntry> = extension_paths
            .iter()
            .filter_map(|path| {
                let resolved = self.resolve_extension_load_path(path);
                preloaded_by_path.get(&resolved).cloned()
            })
            .collect();
        ordered_extensions.extend(inline_extensions);

        ExtensionLoadResult {
            extensions: ordered_extensions,
            errors: pre_trust
                .errors
                .iter()
                .chain(remaining_extensions.errors.iter())
                .cloned()
                .collect(),
            runtime: pre_trust.runtime.clone(),
        }
    }

    /// The extension-load seam call, upstream's `loadExtensionsCached`.
    async fn call_load_extensions(&self, extension_paths: &[String]) -> ExtensionLoadResult {
        (self.load_extensions_fn)(extension_paths.to_vec(), self.cwd.clone()).await
    }

    /// The extension load path resolver, upstream's
    /// `resolveExtensionLoadPath`: cwd-based with unicode space variants
    /// normalized.
    fn resolve_extension_load_path(&self, path: &str) -> String {
        resolve_path_with(
            path,
            &self.cwd,
            &PathInputOptions {
                normalize_unicode_spaces: true,
                ..PathInputOptions::default()
            },
        )
        .unwrap_or_else(|_| path.to_string())
    }

    /// Map an auto-discovered or package skill resource to its SKILL.md
    /// file when the resource is a directory carrying one, upstream's
    /// `mapSkillPath`; the metadata rides the mapped file.
    fn map_skill_path(
        resource: &ResolvedResource,
        metadata_by_path: &mut HashMap<String, PathMetadata>,
    ) -> String {
        if resource.metadata.source != "auto" && resource.metadata.origin != SourceOrigin::Package {
            return resource.path.clone();
        }
        let Ok(stats) = std::fs::metadata(&resource.path) else {
            return resource.path.clone();
        };
        if !stats.is_dir() {
            return resource.path.clone();
        }
        let skill_file = Path::new(&resource.path)
            .join("SKILL.md")
            .to_string_lossy()
            .into_owned();
        if Path::new(&skill_file).exists() {
            metadata_by_path
                .entry(skill_file.clone())
                .or_insert_with(|| resource.metadata.clone());
            return skill_file;
        }
        resource.path.clone()
    }

    /// Normalize extension-supplied resource entries, upstream's
    /// `normalizeExtensionPaths`: paths and base dirs resolve against the
    /// cwd.
    fn normalize_extension_paths(&self, entries: &[PathWithMetadata]) -> Vec<PathWithMetadata> {
        entries
            .iter()
            .map(|entry| {
                let metadata = PathMetadata {
                    base_dir: entry
                        .metadata
                        .base_dir
                        .as_ref()
                        .map(|base_dir| self.resolve_resource_path(base_dir)),
                    ..entry.metadata.clone()
                };
                PathWithMetadata {
                    path: self.resolve_resource_path(&entry.path),
                    metadata,
                }
            })
            .collect()
    }

    /// Reload the skill set from paths, upstream's
    /// `updateSkillsFromPaths`, stamping each skill's provenance from the
    /// extension metadata, the skill's own loader attribution, or the
    /// default path attribution.
    fn update_skills_from_paths(
        &mut self,
        skill_paths: &[String],
        metadata_by_path: Option<&HashMap<String, PathMetadata>>,
    ) {
        let skills_result = if self.no_skills && skill_paths.is_empty() {
            LoadedSkillsResult::default()
        } else {
            let loaded = load_skills(&crate::skills::LoadSkillsOptions {
                cwd: &self.cwd,
                agent_dir: &self.agent_dir,
                skill_paths,
                include_defaults: false,
            });
            LoadedSkillsResult {
                skills: loaded.skills,
                diagnostics: loaded.diagnostics,
            }
        };
        let resolved_skills = match &self.skills_override {
            Some(skills_override) => skills_override(skills_result),
            None => skills_result,
        };
        let extension_skill_source_infos = self.extension_skill_source_infos.clone();
        self.skills = resolved_skills
            .skills
            .into_iter()
            .map(|mut skill| {
                skill.source_info = self
                    .find_source_info_for_path(
                        &skill.file_path,
                        Some(&extension_skill_source_infos),
                        metadata_by_path,
                    )
                    .or(Some(skill.source_info))
                    .unwrap_or_else(|| self.get_default_source_info_for_path(&skill.file_path));
                skill
            })
            .collect();
        self.skill_diagnostics = resolved_skills.diagnostics;
    }

    /// Reload the prompt set from paths, upstream's
    /// `updatePromptsFromPaths` with the name dedupe.
    fn update_prompts_from_paths(
        &mut self,
        prompt_paths: &[String],
        metadata_by_path: Option<&HashMap<String, PathMetadata>>,
    ) {
        let prompts_result = if self.no_prompt_templates && prompt_paths.is_empty() {
            LoadedPromptsResult::default()
        } else {
            let all_prompts =
                load_prompt_templates(&crate::prompt_templates::LoadPromptTemplatesOptions {
                    cwd: &self.cwd,
                    agent_dir: &self.agent_dir,
                    prompt_paths,
                    include_defaults: false,
                });
            let (prompts, diagnostics) = dedupe_prompts(all_prompts);
            LoadedPromptsResult {
                prompts,
                diagnostics,
            }
        };
        let resolved_prompts = match &self.prompts_override {
            Some(prompts_override) => prompts_override(prompts_result),
            None => prompts_result,
        };
        let extension_prompt_source_infos = self.extension_prompt_source_infos.clone();
        self.prompts = resolved_prompts
            .prompts
            .into_iter()
            .map(|mut prompt| {
                prompt.source_info = self
                    .find_source_info_for_path(
                        &prompt.file_path,
                        Some(&extension_prompt_source_infos),
                        metadata_by_path,
                    )
                    .or(Some(prompt.source_info))
                    .unwrap_or_else(|| self.get_default_source_info_for_path(&prompt.file_path));
                prompt
            })
            .collect();
        self.prompt_diagnostics = resolved_prompts.diagnostics;
    }

    /// Reload the theme set from paths, upstream's
    /// `updateThemesFromPaths` with the name dedupe.
    fn update_themes_from_paths(
        &mut self,
        theme_paths: &[String],
        metadata_by_path: Option<&HashMap<String, PathMetadata>>,
    ) {
        let themes_result = if self.no_themes && theme_paths.is_empty() {
            LoadedThemesResult::default()
        } else {
            let loaded = self.load_themes(theme_paths);
            let (themes, dedupe_diagnostics) = dedupe_themes(loaded.themes);
            let mut diagnostics = loaded.diagnostics;
            diagnostics.extend(dedupe_diagnostics);
            LoadedThemesResult {
                themes,
                diagnostics,
            }
        };
        let resolved_themes = match &self.themes_override {
            Some(themes_override) => themes_override(themes_result),
            None => themes_result,
        };
        let extension_theme_source_infos = self.extension_theme_source_infos.clone();
        self.themes = resolved_themes
            .themes
            .into_iter()
            .map(|mut theme| {
                if let Some(source_path) = theme.source_path.clone() {
                    theme.source_info = self
                        .find_source_info_for_path(
                            &source_path,
                            Some(&extension_theme_source_infos),
                            metadata_by_path,
                        )
                        .or(theme.source_info)
                        .or_else(|| Some(self.get_default_source_info_for_path(&source_path)));
                }
                theme
            })
            .collect();
        self.theme_diagnostics = resolved_themes.diagnostics;
    }

    /// The provenance for a resource path, upstream's
    /// `findSourceInfoForPath`: an extension's recorded metadata wins, then
    /// the resolved path metadata (exact, then by directory prefix).
    fn find_source_info_for_path(
        &self,
        resource_path: &str,
        extra_source_infos: Option<&HashMap<String, SourceInfo>>,
        metadata_by_path: Option<&HashMap<String, PathMetadata>>,
    ) -> Option<SourceInfo> {
        if resource_path.is_empty() {
            return None;
        }

        if resource_path.starts_with('<') {
            return Some(self.get_default_source_info_for_path(resource_path));
        }

        let normalized_resource_path = resolve_default(resource_path);
        if let Some(extra_source_infos) = extra_source_infos {
            for (source_path, source_info) in extra_source_infos {
                let normalized_source_path = resolve_default(source_path);
                if normalized_resource_path == normalized_source_path
                    || normalized_resource_path.starts_with(&format!("{normalized_source_path}/"))
                {
                    return Some(SourceInfo {
                        path: resource_path.to_string(),
                        ..source_info.clone()
                    });
                }
            }
        }

        if let Some(metadata_by_path) = metadata_by_path {
            if let Some(exact) = metadata_by_path
                .get(&normalized_resource_path)
                .or_else(|| metadata_by_path.get(resource_path))
            {
                return Some(create_source_info(resource_path, exact));
            }

            for (source_path, metadata) in metadata_by_path {
                let normalized_source_path = resolve_default(source_path);
                if normalized_resource_path == normalized_source_path
                    || normalized_resource_path.starts_with(&format!("{normalized_source_path}/"))
                {
                    return Some(create_source_info(resource_path, metadata));
                }
            }
        }

        None
    }

    /// The default provenance for a resource path, upstream's
    /// `getDefaultSourceInfoForPath`: the agent dir's resource trees read
    /// user, the project `.pi` trees read project, everything else reads
    /// temporary with its containing directory as the base.
    fn get_default_source_info_for_path(&self, file_path: &str) -> SourceInfo {
        if file_path.starts_with('<') && file_path.ends_with('>') {
            let inner = &file_path[1..file_path.len().saturating_sub(1)];
            let source = inner.split(':').next().unwrap_or_default();
            return SourceInfo {
                path: file_path.to_string(),
                source: if source.is_empty() {
                    "temporary".to_string()
                } else {
                    source.to_string()
                },
                scope: SourceScope::Temporary,
                origin: SourceOrigin::TopLevel,
                base_dir: None,
            };
        }

        let normalized_path = resolve_default(file_path);
        let agent_roots = ["skills", "prompts", "themes", "extensions"]
            .iter()
            .map(|name| {
                Path::new(&self.agent_dir)
                    .join(name)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        let project_roots = ["skills", "prompts", "themes", "extensions"]
            .iter()
            .map(|name| {
                Path::new(&self.cwd)
                    .join(CONFIG_DIR_NAME)
                    .join(name)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();

        for root in &agent_roots {
            if is_under_path(&normalized_path, root) {
                return SourceInfo {
                    path: file_path.to_string(),
                    source: "local".to_string(),
                    scope: SourceScope::User,
                    origin: SourceOrigin::TopLevel,
                    base_dir: Some(root.clone()),
                };
            }
        }

        for root in &project_roots {
            if is_under_path(&normalized_path, root) {
                return SourceInfo {
                    path: file_path.to_string(),
                    source: "local".to_string(),
                    scope: SourceScope::Project,
                    origin: SourceOrigin::TopLevel,
                    base_dir: Some(root.clone()),
                };
            }
        }

        let is_directory = std::fs::metadata(&normalized_path).is_ok_and(|stats| stats.is_dir());
        SourceInfo {
            path: file_path.to_string(),
            source: "local".to_string(),
            scope: SourceScope::Temporary,
            origin: SourceOrigin::TopLevel,
            base_dir: Some(if is_directory {
                normalized_path
            } else {
                dirname_posix_loader(&normalized_path)
            }),
        }
    }

    /// Merge path lists, deduplicating by canonical path while keeping
    /// first position, upstream's `mergePaths`.
    fn merge_paths(&self, primary: &[String], additional: &[String]) -> Vec<String> {
        let mut merged: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for p in primary.iter().chain(additional.iter()) {
            let resolved = self.resolve_resource_path(p);
            let canonical_path = canonicalize_path(&resolved);
            if seen.contains(&canonical_path) {
                continue;
            }
            seen.insert(canonical_path);
            merged.push(resolved);
        }

        merged
    }

    /// The resource path resolver, upstream's `resolveResourcePath`:
    /// cwd-based with whitespace trimmed.
    fn resolve_resource_path(&self, p: &str) -> String {
        resolve_path_with(
            p,
            &self.cwd,
            &PathInputOptions {
                trim: true,
                ..PathInputOptions::default()
            },
        )
        .unwrap_or_else(|_| p.to_string())
    }

    /// Load theme files and directories, upstream's `loadThemes`.
    ///
    /// Upstream's `includeDefaults` arm — a caller-controlled default-dir
    /// scan — has no caller at the pin (the loader always passes false),
    /// so the port drops the parameter with the dead branch.
    #[expect(
        clippy::case_sensitive_file_extension_comparisons,
        reason = "upstream's json ends-with probe is case-sensitive; the 1:1 shape keeps it"
    )]
    fn load_themes(&self, paths: &[String]) -> LoadedThemesResult {
        let mut themes: Vec<ThemeHandle> = Vec::new();
        let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();

        for p in paths {
            let resolved = self.resolve_resource_path(p);
            if !Path::new(&resolved).exists() {
                diagnostics.push(ResourceDiagnostic {
                    kind: ResourceDiagnosticKind::Warning,
                    message: "theme path does not exist".to_string(),
                    path: Some(resolved.clone()),
                    collision: None,
                });
                continue;
            }

            let Ok(stats) = std::fs::metadata(&resolved) else {
                diagnostics.push(ResourceDiagnostic {
                    kind: ResourceDiagnosticKind::Warning,
                    message: "failed to read theme path".to_string(),
                    path: Some(resolved.clone()),
                    collision: None,
                });
                continue;
            };
            if stats.is_dir() {
                self.load_themes_from_dir(&resolved, &mut themes, &mut diagnostics);
            } else if stats.is_file() && resolved.ends_with(".json") {
                self.load_theme_from_file(&resolved, &mut themes, &mut diagnostics);
            } else {
                diagnostics.push(ResourceDiagnostic {
                    kind: ResourceDiagnosticKind::Warning,
                    message: "theme path is not a json file".to_string(),
                    path: Some(resolved.clone()),
                    collision: None,
                });
            }
        }

        LoadedThemesResult {
            themes,
            diagnostics,
        }
    }

    /// Load a theme directory's `.json` children, upstream's
    /// `loadThemesFromDir`; symlinked children stat through.
    #[expect(
        clippy::case_sensitive_file_extension_comparisons,
        reason = "upstream's json ends-with probe is case-sensitive; the 1:1 shape keeps it"
    )]
    fn load_themes_from_dir(
        &self,
        dir: &str,
        themes: &mut Vec<ThemeHandle>,
        diagnostics: &mut Vec<ResourceDiagnostic>,
    ) {
        if !Path::new(dir).exists() {
            return;
        }

        let Ok(read) = std::fs::read_dir(dir) else {
            diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Warning,
                message: "failed to read theme directory".to_string(),
                path: Some(dir.to_string()),
                collision: None,
            });
            return;
        };

        for entry in read.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let mut is_file = file_type.is_file();
            if file_type.is_symlink() {
                let full_path = entry.path();
                match std::fs::metadata(&full_path) {
                    Ok(stats) => is_file = stats.is_file(),
                    Err(_) => continue,
                }
            }
            if !is_file || !name.ends_with(".json") {
                continue;
            }
            let file_path = entry.path().to_string_lossy().into_owned();
            self.load_theme_from_file(&file_path, themes, diagnostics);
        }
    }

    /// Load one theme file through the seam, upstream's
    /// `loadThemeFromFile`: a failure becomes a warning diagnostic and the
    /// theme loads as nothing.
    fn load_theme_from_file(
        &self,
        file_path: &str,
        themes: &mut Vec<ThemeHandle>,
        diagnostics: &mut Vec<ResourceDiagnostic>,
    ) {
        match (self.theme_load_fn)(file_path) {
            Ok(name) => themes.push(ThemeHandle {
                name,
                source_path: Some(file_path.to_string()),
                source_info: None,
            }),
            Err(message) => diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Warning,
                message,
                path: Some(file_path.to_string()),
                collision: None,
            }),
        }
    }

    /// Discover the system prompt file, upstream's
    /// `discoverSystemPromptFile`: the trusted project
    /// `.pi/SYSTEM.md` wins, else the global `SYSTEM.md`.
    fn discover_system_prompt_file(&self) -> Option<String> {
        let project_path = Path::new(&self.cwd).join(CONFIG_DIR_NAME).join("SYSTEM.md");
        let project_trusted = self
            .settings_manager
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_project_trusted();
        if project_trusted && project_path.exists() {
            return Some(project_path.to_string_lossy().into_owned());
        }

        let global_path = Path::new(&self.agent_dir).join("SYSTEM.md");
        if global_path.exists() {
            return Some(global_path.to_string_lossy().into_owned());
        }

        None
    }

    /// Discover the append system prompt file, upstream's
    /// `discoverAppendSystemPromptFile`: the trusted project
    /// `.pi/APPEND_SYSTEM.md` wins, else the global file.
    fn discover_append_system_prompt_file(&self) -> Option<String> {
        let project_path = Path::new(&self.cwd)
            .join(CONFIG_DIR_NAME)
            .join("APPEND_SYSTEM.md");
        let project_trusted = self
            .settings_manager
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_project_trusted();
        if project_trusted && project_path.exists() {
            return Some(project_path.to_string_lossy().into_owned());
        }

        let global_path = Path::new(&self.agent_dir).join("APPEND_SYSTEM.md");
        if global_path.exists() {
            return Some(global_path.to_string_lossy().into_owned());
        }

        None
    }
}

impl<S: SettingsStorage + 'static> DefaultResourceLoader<S> {
    /// The loader with the file-backed settings manager built from the
    /// cwd and agent dir, upstream's `SettingsManager.create` default in
    /// the options object.
    #[must_use]
    pub fn with_default_settings(
        options: DefaultResourceLoaderOptions,
    ) -> DefaultResourceLoader<FileSettingsStorage> {
        let cwd = resolve_default(&options.cwd);
        let agent_dir = resolve_default(&options.agent_dir);
        let settings_manager = SettingsManager::create(
            &cwd,
            &agent_dir,
            crate::settings_manager::SettingsManagerCreateOptions::default(),
        );
        DefaultResourceLoader::new(options, settings_manager)
    }
}

impl<S: SettingsStorage> std::fmt::Debug for DefaultResourceLoader<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultResourceLoader")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .field("extensions_result", &self.extensions_result)
            .field("skill_count", &self.skills.len())
            .field("prompt_count", &self.prompts.len())
            .field("theme_count", &self.themes.len())
            .field("agents_file_count", &self.agents_files.len())
            .field("system_prompt_set", &self.system_prompt.is_some())
            .finish_non_exhaustive()
    }
}

/// The default extension-load seam until #128: every pass resolves to an
/// empty result, the documented degradation of upstream's module loading.
fn default_load_extensions_fn() -> LoadExtensionsFn {
    Arc::new(|_extension_paths: Vec<String>, _cwd: String| {
        Box::pin(async { ExtensionLoadResult::default() })
    })
}

/// Dedupe prompts by name, upstream's `dedupePrompts`: first wins, later
/// same-name prompts record a collision diagnostic.
fn dedupe_prompts(prompts: Vec<PromptTemplate>) -> (Vec<PromptTemplate>, Vec<ResourceDiagnostic>) {
    let mut seen: IndexMap<String, PromptTemplate> = IndexMap::new();
    let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();

    for prompt in prompts {
        if let Some(existing) = seen.get(&prompt.name) {
            diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Collision,
                message: format!("name \"/{}\" collision", prompt.name),
                path: Some(prompt.file_path.clone()),
                collision: Some(ResourceCollision {
                    resource_type: ResourceType::Prompt,
                    name: prompt.name.clone(),
                    winner_path: existing.file_path.clone(),
                    loser_path: prompt.file_path.clone(),
                    winner_source: None,
                    loser_source: None,
                }),
            });
        } else {
            seen.insert(prompt.name.clone(), prompt);
        }
    }

    (seen.into_values().collect(), diagnostics)
}

/// Dedupe themes by name, upstream's `dedupeThemes`: first wins, later
/// same-name themes record a collision; unnamed themes dedupe as
/// `unnamed`.
fn dedupe_themes(themes: Vec<ThemeHandle>) -> (Vec<ThemeHandle>, Vec<ResourceDiagnostic>) {
    let mut seen: IndexMap<String, ThemeHandle> = IndexMap::new();
    let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();

    for theme in themes {
        let name = theme.name.clone().unwrap_or_else(|| "unnamed".to_string());
        if let Some(existing) = seen.get(&name) {
            diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Collision,
                message: format!("name \"{name}\" collision"),
                path: theme.source_path.clone(),
                collision: Some(ResourceCollision {
                    resource_type: ResourceType::Theme,
                    name,
                    winner_path: existing
                        .source_path
                        .clone()
                        .unwrap_or_else(|| "<builtin>".to_string()),
                    loser_path: theme
                        .source_path
                        .clone()
                        .unwrap_or_else(|| "<builtin>".to_string()),
                    winner_source: None,
                    loser_source: None,
                }),
            });
        } else {
            seen.insert(name, theme);
        }
    }

    (seen.into_values().collect(), diagnostics)
}

/// The resolver defaults for the loader's own paths, upstream's
/// `resolvePath(input)` with no base.
fn resolve_default(input: &str) -> String {
    let base = crate::config::process_cwd();
    resolve_path_with(
        input,
        &base,
        &PathInputOptions {
            home_dir: Some(crate::config::home_dir()),
            ..PathInputOptions::default()
        },
    )
    .unwrap_or_else(|_| input.to_string())
}

/// The dirname over POSIX separators, Node's `path.dirname`.
fn dirname_posix_loader(p: &str) -> String {
    let trimmed = p.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(0) => "/".to_string(),
        Some(at) => trimmed[..at].to_string(),
        None => ".".to_string(),
    }
}

/// The parent of a POSIX path, `None` at the root — upstream's
/// `dirname(currentDir) === currentDir` loop terminator.
fn parent_dir_posix(dir: &str) -> Option<String> {
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

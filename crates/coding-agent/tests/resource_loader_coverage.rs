//! Coverage-closing cases for the resource loader at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the branches the 1:1 and
//! boundary suites leave open — the debug surfaces, the filesystem
//! warning arms upstream answers with `console.error`, the extension
//! seam's preload/remaining pass with its synthetic labels, the default
//! provenance ladder, the theme read failures, and the git-metadata
//! probe's failure paths.
//!
//! Deliberately uncovered, mirroring upstream's reachability: the
//! reload-level "Skill path does not exist" and "Theme path does not
//! exist" error pushes (the inner loaders pre-record the same paths as
//! warnings, so the guard never opens — the prompt push is the live one
//! and the boundary suite covers it), the CLI skills metadata insert
//! (`resolveExtensionSources` fills extensions only, so upstream's
//! skills loop never runs), the `mapSkillPath` directory arms (both
//! collectors resolve to files; the directory mapping is defensive),
//! and the stat-failure arms that only a filesystem race reaches.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pi_coding_agent::diagnostics::ResourceDiagnosticKind;
use pi_coding_agent::footer_data_provider::find_git_paths;
use pi_coding_agent::package_manager::PathMetadata;
use pi_coding_agent::resource_loader::{
    ContextFile, DefaultResourceLoader, DefaultResourceLoaderOptions, ExtensionEntry,
    ExtensionLoadError, ExtensionLoadResult, ExtensionRuntimeHandle, LoadedPromptsResult,
    LoadedThemesResult, PathWithMetadata, ResourceExtensionPaths, ResourceLoaderReloadOptions,
    ThemeHandle,
};
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
};
use pi_coding_agent::source_info::{SourceOrigin, SourceScope};

struct LoaderEnv {
    root: PathBuf,
    agent_dir: String,
    cwd: String,
}

impl LoaderEnv {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("tempdir").keep();
        let agent_dir = root.join("agent");
        let cwd = root.join("project");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        std::fs::create_dir_all(&cwd).expect("cwd");
        Self {
            agent_dir: agent_dir.to_string_lossy().into_owned(),
            cwd: cwd.to_string_lossy().into_owned(),
            root,
        }
    }

    fn options(&self) -> DefaultResourceLoaderOptions {
        DefaultResourceLoaderOptions {
            cwd: self.cwd.clone(),
            agent_dir: self.agent_dir.clone(),
            ..DefaultResourceLoaderOptions::default()
        }
    }

    fn write(&self, relative: &str, content: &str) -> String {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&path, content).expect("write");
        path.to_string_lossy().into_owned()
    }
}

fn in_memory_manager() -> SettingsManager<InMemorySettingsStorage> {
    SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default())
}

fn untrusted_in_memory_manager() -> SettingsManager<InMemorySettingsStorage> {
    SettingsManager::in_memory(
        &Settings::new(),
        SettingsManagerCreateOptions {
            project_trusted: Some(false),
        },
    )
}

fn make_unreadable(path: &Path) {
    let mut permissions = std::fs::metadata(path).expect("stat").permissions();
    permissions.set_mode(0o000);
    std::fs::set_permissions(path, permissions).expect("chmod");
}

fn restore_permissions(path: &Path) {
    let mut permissions = std::fs::metadata(path).expect("stat").permissions();
    permissions.set_mode(0o644);
    std::fs::set_permissions(path, permissions).expect("chmod");
}

fn extension_entry(path: &str) -> ExtensionEntry {
    ExtensionEntry {
        path: path.to_string(),
        resolved_path: path.to_string(),
        hidden: false,
        source_info: None,
    }
}

// === debug surfaces =========================================================

#[test]
fn debug_impls_render_the_option_and_loader_surfaces() {
    let env = LoaderEnv::new();
    let options = DefaultResourceLoaderOptions {
        cwd: env.cwd.clone(),
        agent_dir: env.agent_dir.clone(),
        additional_extension_paths: vec!["/tmp/x.ts".to_string()],
        no_extensions: true,
        system_prompt: Some("be brief".to_string()),
        ..DefaultResourceLoaderOptions::default()
    };
    let rendered = format!("{options:?}");
    assert!(rendered.contains("DefaultResourceLoaderOptions"));
    assert!(rendered.contains("no_extensions: true"));

    let reload = ResourceLoaderReloadOptions {
        resolve_project_trust: Some(Box::new(|_: ExtensionLoadResult| Box::pin(async { true }))),
    };
    let rendered = format!("{reload:?}");
    assert!(rendered.contains("ResourceLoaderReloadOptions"));
    assert!(rendered.contains("<callback>"));

    let loader =
        DefaultResourceLoader::<pi_coding_agent::settings_manager::FileSettingsStorage>::with_default_settings(
            env.options(),
        );
    let rendered = format!("{loader:?}");
    assert!(rendered.contains("DefaultResourceLoader"));
    assert!(rendered.contains("system_prompt_set: false"));
}

// === filesystem warning arms ===============================================

#[tokio::test]
async fn unreadable_prompt_and_context_files_warn_and_fall_back() {
    let env = LoaderEnv::new();
    let system_md = env.write("agent/SYSTEM.md", "real prompt");
    make_unreadable(Path::new(&system_md));
    let agents_md = env.write("project/AGENTS.md", "project context");
    make_unreadable(Path::new(&agents_md));

    let mut loader = DefaultResourceLoader::new(env.options(), in_memory_manager());
    loader.reload(None).await;

    // The system prompt read failure warns and falls back to the literal
    // source text — the path string itself, upstream's console.error +
    // literal return.
    assert_eq!(
        loader.get_system_prompt().as_deref(),
        Some(system_md.as_str())
    );
    // The context file read failure warns and moves on: the project dir
    // contributes no context file.
    assert!(loader.get_agents_files().is_empty());
}

// === extension seam =========================================================

#[tokio::test]
async fn extension_seam_results_flow_through_preload_and_remaining_paths() {
    let env = LoaderEnv::new();
    let path_a = env.write("ext/a.ts", "");
    let path_b = env.write("ext/b.ts", "");
    let path_c = env.write("ext/c.ts", "");

    let mut options = env.options();
    options.additional_extension_paths = vec![path_a.clone(), path_b.clone(), path_c.clone()];
    let calls: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&calls);
    let (closure_a, closure_b, closure_c) = (path_a.clone(), path_b.clone(), path_c.clone());
    options.load_extensions_fn = Some(Arc::new(move |paths: Vec<String>, _cwd: String| {
        recorded.lock().expect("calls").push(paths);
        let first = recorded.lock().expect("calls").len() == 1;
        let path_a = closure_a.clone();
        let path_b = closure_b.clone();
        let path_c = closure_c.clone();
        Box::pin(async move {
            if first {
                // The pre-trust bootstrap: preload A, fail B, leave C for
                // the trusted pass, and contribute one inline factory.
                ExtensionLoadResult {
                    extensions: vec![
                        extension_entry(&path_a),
                        ExtensionEntry {
                            path: "<inline:1>".to_string(),
                            resolved_path: "<inline:1>".to_string(),
                            hidden: true,
                            source_info: None,
                        },
                    ],
                    errors: vec![ExtensionLoadError {
                        path: path_b.clone(),
                        error: "boom".to_string(),
                    }],
                    runtime: ExtensionRuntimeHandle,
                }
            } else {
                // The trusted pass loads the remaining path.
                ExtensionLoadResult {
                    extensions: vec![extension_entry(&path_c)],
                    errors: vec![],
                    runtime: ExtensionRuntimeHandle,
                }
            }
        })
    }));

    let mut loader = DefaultResourceLoader::new(options, in_memory_manager());
    loader
        .reload(Some(ResourceLoaderReloadOptions {
            resolve_project_trust: Some(Box::new(|_: ExtensionLoadResult| {
                Box::pin(async { true })
            })),
        }))
        .await;

    let result = loader.get_extensions();
    // Path-backed entries order by the requested paths (A then C); the
    // inline factory appends last.
    let paths: Vec<&str> = result
        .extensions
        .iter()
        .map(|entry| entry.path.as_str())
        .collect();
    assert_eq!(paths, vec![path_a.as_str(), path_c.as_str(), "<inline:1>"]);
    // The failed pre-trust path stays an error; the second pass adds none.
    assert_eq!(
        result.errors,
        vec![ExtensionLoadError {
            path: path_b.clone(),
            error: "boom".to_string(),
        }]
    );
    // The synthetic label reads the inline prefix; the CLI-resolved source
    // stamps A and C through the exact metadata match.
    let inline = result
        .extensions
        .iter()
        .find(|entry| entry.path == "<inline:1>")
        .expect("inline entry");
    let source_info = inline.source_info.as_ref().expect("stamped");
    assert_eq!(source_info.source, "inline");
    assert!(matches!(source_info.scope, SourceScope::Temporary));
    for path in [&path_a, &path_c] {
        let entry = result
            .extensions
            .iter()
            .find(|entry| entry.path == *path)
            .expect("stamped entry");
        assert_eq!(entry.source_info.as_ref().expect("stamped").source, "cli");
    }
    // Two seam calls: the pre-trust bootstrap and the trusted remainder.
    let calls = calls.lock().expect("calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0],
        vec![path_a.clone(), path_b.clone(), path_c.clone()]
    );
    assert_eq!(calls[1], vec![path_c.clone()]);
}

#[tokio::test]
async fn package_skill_and_extension_resources_stamp_their_package_metadata() {
    let env = LoaderEnv::new();
    let package_root = env.root.join("pkg");
    std::fs::create_dir_all(package_root.join("skills")).expect("mkdir");
    std::fs::create_dir_all(package_root.join("extensions")).expect("mkdir");
    let skill_md = env.write(
        "pkg/skills/SKILL.md",
        "---\nname: pkg-skill\ndescription: from the package\n---\nbody",
    );
    let extension_ts = env.write("pkg/extensions/tool.ts", "");

    let mut manager = in_memory_manager();
    manager.set_packages(&[serde_json::Value::String(
        package_root.to_string_lossy().into_owned(),
    )]);

    let mut options = env.options();
    let extension_ts_clone = extension_ts.clone();
    options.load_extensions_fn = Some(Arc::new(move |paths: Vec<String>, _cwd: String| {
        let extension_ts = extension_ts_clone.clone();
        Box::pin(async move {
            ExtensionLoadResult {
                extensions: paths
                    .iter()
                    .filter(|path| path.as_str() == extension_ts)
                    .map(|path| extension_entry(path))
                    .collect(),
                errors: vec![],
                runtime: ExtensionRuntimeHandle,
            }
        })
    }));

    let mut loader = DefaultResourceLoader::new(options, manager);
    loader.reload(None).await;

    let skills = loader.get_skills().skills;
    let skill = skills
        .iter()
        .find(|skill| skill.name == "pkg-skill")
        .expect("package skill");
    assert_eq!(
        skill.source_info.source,
        package_root.to_string_lossy().to_string()
    );
    assert!(matches!(skill.source_info.scope, SourceScope::User));
    assert!(matches!(skill.source_info.origin, SourceOrigin::Package));

    let result = loader.get_extensions();
    let entry = result
        .extensions
        .iter()
        .find(|entry| entry.path == extension_ts)
        .expect("package extension");
    let source_info = entry.source_info.as_ref().expect("stamped");
    assert_eq!(source_info.source, package_root.to_string_lossy());
    assert_eq!(skill_md, skill.file_path);
}

#[tokio::test]
async fn default_provenance_ladder_reads_agent_project_and_temporary_roots() {
    let env = LoaderEnv::new();
    // Theme files inside the auto roots but below their direct children
    // stay uncollected (the auto theme reader reads direct `.json`
    // children only), and CLI theme paths carry no metadata at all — the
    // theme stamp's `or_else` falls through the default ladder.
    let user_theme = env.write(
        "agent/themes/nested/cli-user.json",
        "{\"name\": \"cli-user\"}",
    );
    let project_theme = env.write(
        "project/.pi/themes/nested/cli-project.json",
        "{\"name\": \"cli-project\"}",
    );
    let temp_theme = env.write("cli-temp.json", "{\"name\": \"cli-temp\"}");

    let mut options = env.options();
    options.additional_theme_paths = vec![
        user_theme.clone(),
        project_theme.clone(),
        temp_theme.clone(),
    ];

    let mut loader = DefaultResourceLoader::new(options, untrusted_in_memory_manager());
    loader.reload(None).await;

    let themes = loader.get_themes().themes;
    let info_of = |name: &str| {
        themes
            .iter()
            .find(|theme| theme.name.as_deref() == Some(name))
            .expect("theme")
            .source_info
            .clone()
            .expect("stamped")
    };
    let user = info_of("cli-user");
    assert!(matches!(user.scope, SourceScope::User));
    assert!(
        user.base_dir
            .as_deref()
            .unwrap_or_default()
            .ends_with("agent/themes")
    );
    let project = info_of("cli-project");
    assert!(matches!(project.scope, SourceScope::Project));
    assert!(
        project
            .base_dir
            .as_deref()
            .unwrap_or_default()
            .ends_with(".pi/themes")
    );
    let temp = info_of("cli-temp");
    assert!(matches!(temp.scope, SourceScope::Temporary));
    assert_eq!(
        temp.base_dir.as_deref(),
        Some(env.root.to_string_lossy().as_ref())
    );
}

// === overrides ==============================================================

#[tokio::test]
async fn overrides_shape_the_loaded_resource_families() {
    let env = LoaderEnv::new();
    env.write("agent/prompts/greet.md", "hello");

    let mut options = env.options();
    options.agents_files_override = Some(Box::new(|files: Vec<ContextFile>| {
        let mut files = files;
        files.push(ContextFile {
            path: "/synthetic/AGENTS.md".to_string(),
            content: "synthetic context".to_string(),
        });
        files
    }));
    options.prompts_override = Some(Box::new(|mut result: LoadedPromptsResult| {
        if let Some(prompt) = result.prompts.first_mut() {
            prompt.content = "overridden".to_string();
        }
        result
    }));
    options.themes_override = Some(Box::new(|mut result: LoadedThemesResult| {
        result.themes.push(ThemeHandle {
            name: Some("synthetic".to_string()),
            source_path: None,
            source_info: None,
        });
        result
    }));
    options.append_system_prompt_override = Some(Box::new(|base: Vec<String>| {
        let mut base = base;
        base.push("synthetic append".to_string());
        base
    }));

    let mut loader = DefaultResourceLoader::new(options, in_memory_manager());
    loader.reload(None).await;

    let prompts = loader.get_prompts().prompts;
    assert_eq!(prompts.first().expect("prompt").content, "overridden");
    let themes = loader.get_themes().themes;
    assert!(
        themes
            .iter()
            .any(|theme| theme.name.as_deref() == Some("synthetic"))
    );
    assert_eq!(
        loader.get_append_system_prompt(),
        vec!["synthetic append".to_string()]
    );
    assert!(
        loader
            .get_agents_files()
            .iter()
            .any(|file| file.path == "/synthetic/AGENTS.md")
    );
}

// === theme read failures ====================================================

#[tokio::test]
async fn unreadable_theme_directory_warns_and_answers_no_themes() {
    let env = LoaderEnv::new();
    let dir = env.root.join("theme-bundle");
    std::fs::create_dir_all(&dir).expect("mkdir");
    make_unreadable(&dir);

    let mut options = env.options();
    options.additional_theme_paths = vec![dir.to_string_lossy().into_owned()];

    let mut loader = DefaultResourceLoader::new(options, in_memory_manager());
    loader.reload(None).await;

    let result = loader.get_themes();
    assert!(result.themes.is_empty());
    assert!(result.diagnostics.iter().any(|d| {
        d.kind == ResourceDiagnosticKind::Warning
            && d.message == "failed to read theme directory"
            && d.path.as_deref() == Some(dir.to_string_lossy().as_ref())
    }));
}

#[tokio::test]
async fn theme_directory_walk_reads_files_skips_subdirectories() {
    let env = LoaderEnv::new();
    let dir = env.root.join("theme-bundle");
    std::fs::create_dir_all(dir.join("nested")).expect("mkdir");
    let theme = env.write("theme-bundle/dark.json", "{\"name\": \"dark\"}");
    let _nested = env.write("theme-bundle/nested/inner.json", "{\"name\": \"inner\"}");

    let mut options = env.options();
    options.additional_theme_paths = vec![dir.to_string_lossy().into_owned()];

    let mut loader = DefaultResourceLoader::new(options, in_memory_manager());
    loader.reload(None).await;

    let themes = loader.get_themes().themes;
    // Direct files load; nested directories are not walked.
    assert_eq!(themes.len(), 1);
    assert_eq!(themes[0].name.as_deref(), Some("dark"));
    assert_eq!(themes[0].source_path.as_deref(), Some(theme.as_str()));
    assert!(matches!(
        themes[0].source_info.as_ref().expect("stamped").scope,
        SourceScope::Temporary
    ));
}

#[tokio::test]
async fn global_system_md_is_discovered_when_the_project_has_none() {
    let env = LoaderEnv::new();
    let system_md = env.write("agent/SYSTEM.md", "global prompt");
    let _append = env.write("agent/APPEND_SYSTEM.md", "global append");

    let mut loader = DefaultResourceLoader::new(env.options(), in_memory_manager());
    loader.reload(None).await;

    assert_eq!(loader.get_system_prompt().as_deref(), Some("global prompt"));
    assert_eq!(
        loader.get_system_prompt_source().map(|source| source.path),
        Some(system_md.clone())
    );
    assert!(
        loader
            .get_append_system_prompt_sources()
            .iter()
            .any(|source| source.path.ends_with("APPEND_SYSTEM.md"))
    );
}

// === the git-metadata probe's failure paths =================================

#[test]
fn find_git_paths_failure_arms_answer_none() {
    let env = tempfile::tempdir().expect("tempdir").keep();
    // A `gitdir:` file whose target carries no HEAD answers None.
    let git_dir = env.join("fake-git");
    std::fs::create_dir_all(&git_dir).expect("mkdir");
    std::fs::write(
        env.join(".git"),
        format!("gitdir: {}", git_dir.to_string_lossy()),
    )
    .expect("write");
    assert!(find_git_paths(env.to_string_lossy().as_ref()).is_none());

    // A `gitdir:` file whose target carries a HEAD but no commondir
    // answers with the target as the common git dir.
    std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/main").expect("write");
    let paths = find_git_paths(env.to_string_lossy().as_ref()).expect("git paths");
    assert_eq!(paths.common_git_dir, git_dir.to_string_lossy());

    // An unreadable commondir answers None; restored, its relative target
    // resolves lexically against the git dir.
    let commondir = git_dir.join("commondir");
    std::fs::write(&commondir, "../main.git").expect("write");
    make_unreadable(&commondir);
    assert!(find_git_paths(env.to_string_lossy().as_ref()).is_none());
    restore_permissions(&commondir);
    let paths = find_git_paths(env.to_string_lossy().as_ref()).expect("git paths");
    assert!(paths.common_git_dir.ends_with("main.git"));

    // An unreadable `.git` file answers None.
    let git_file = env.join(".git");
    make_unreadable(&git_file);
    assert!(find_git_paths(env.to_string_lossy().as_ref()).is_none());
    restore_permissions(&git_file);

    // A `.git` directory without a HEAD answers None.
    std::fs::remove_file(&git_file).expect("remove");
    std::fs::create_dir_all(&git_file).expect("mkdir");
    assert!(find_git_paths(env.to_string_lossy().as_ref()).is_none());
}

// === extension-supplied resource paths ======================================

#[tokio::test]
async fn extend_resources_merge_into_the_loader_and_stamp_metadata() {
    let env = LoaderEnv::new();
    let skill_md = env.write(
        "ext-skill/SKILL.md",
        "---\nname: ext-skill\ndescription: from the extension\n---\nbody",
    );
    let prompt_md = env.write("ext-prompt/p.md", "ext prompt");
    let theme_json = env.write("ext-theme/t.json", "{\"name\": \"ext-theme\"}");

    let mut loader = DefaultResourceLoader::new(env.options(), in_memory_manager());
    loader.reload(None).await;

    let metadata = PathMetadata {
        source: "ext-package".to_string(),
        scope: SourceScope::User,
        origin: SourceOrigin::Package,
        base_dir: None,
    };
    loader.extend_resources(&ResourceExtensionPaths {
        skill_paths: vec![PathWithMetadata {
            path: skill_md.clone(),
            metadata: metadata.clone(),
        }],
        prompt_paths: vec![PathWithMetadata {
            path: prompt_md.clone(),
            metadata: metadata.clone(),
        }],
        theme_paths: vec![PathWithMetadata {
            path: theme_json.clone(),
            metadata,
        }],
    });

    let skills = loader.get_skills().skills;
    let skill = skills
        .iter()
        .find(|skill| skill.name == "ext-skill")
        .expect("skill");
    assert_eq!(skill.source_info.source, "ext-package");
    let prompts = loader.get_prompts().prompts;
    let prompt = prompts
        .iter()
        .find(|prompt| prompt.name == "p")
        .expect("prompt");
    assert_eq!(prompt.content, "ext prompt");
    let themes = loader.get_themes().themes;
    let theme = themes
        .iter()
        .find(|theme| theme.name.as_deref() == Some("ext-theme"))
        .expect("theme");
    assert_eq!(
        theme.source_info.as_ref().expect("stamped").source,
        "ext-package"
    );
}

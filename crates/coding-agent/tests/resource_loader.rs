//! Upstream `test/resource-loader.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.
//!
//! The suite splits at the port seams. The layering cases — discovery,
//! project-over-user collision preference, context files with the
//! linked-worktree dedup, the `SYSTEM.md`/`APPEND_SYSTEM.md` sources,
//! extension-supplied resources, the no-skip flags, and the overrides —
//! port here. The cases riding other tickets defer: extension loading and
//! its trust bootstrap and conflict detection (#128), the npm package
//! metadata case (#129), and the theme-parser cases (the upstream fixture
//! builds from the interactive theme system's `dark.json`; the theme seam
//! rides #132, so this suite's theme cases restate onto minimal
//! `{"name": …}` JSON files, which carry the same collision substance).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::path::{Path, PathBuf};

use pi_coding_agent::package_manager::PathMetadata;
use pi_coding_agent::resource_loader::PathWithMetadata;
use pi_coding_agent::resource_loader::{
    ContextFile, DefaultResourceLoader, DefaultResourceLoaderOptions, PromptSource,
    ResourceExtensionPaths,
};
use pi_coding_agent::settings_manager::{Settings, SettingsManager, SettingsManagerCreateOptions};
use pi_coding_agent::skills::Skill;
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

    fn write(&self, relative: &str, content: &str) {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(path, content).expect("write");
    }
}

fn file_manager(
    env: &LoaderEnv,
) -> SettingsManager<pi_coding_agent::settings_manager::FileSettingsStorage> {
    SettingsManager::create(
        &env.cwd,
        &env.agent_dir,
        SettingsManagerCreateOptions::default(),
    )
}

fn untrusted_manager(
    env: &LoaderEnv,
) -> SettingsManager<pi_coding_agent::settings_manager::FileSettingsStorage> {
    SettingsManager::create(
        &env.cwd,
        &env.agent_dir,
        SettingsManagerCreateOptions {
            project_trusted: Some(false),
        },
    )
}

fn skill_fixture(name: &str, description: &str) -> String {
    format!("---\nname: {name}\ndescription: {description}\n---\nSkill content here.")
}

// === reload =================================================================

#[tokio::test]
async fn initializes_with_empty_results_before_reload() {
    let env = LoaderEnv::new();
    let loader = DefaultResourceLoader::new(env.options(), file_manager(&env));

    assert_eq!(loader.get_extensions().extensions.len(), 0);
    assert_eq!(loader.get_skills().skills.len(), 0);
    assert_eq!(loader.get_prompts().prompts.len(), 0);
    assert_eq!(loader.get_themes().themes.len(), 0);
}

#[tokio::test]
async fn discovers_skills_from_agent_dir() {
    let env = LoaderEnv::new();
    env.write(
        "agent/skills/test-skill.md",
        &skill_fixture("test-skill", "A test skill"),
    );

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    let skills = loader.get_skills();
    assert!(skills.skills.iter().any(|s| s.name == "test-skill"));
}

#[tokio::test]
async fn ignores_extra_markdown_files_in_auto_discovered_skill_dirs() {
    let env = LoaderEnv::new();
    env.write(
        "agent/skills/pi-skills/browser-tools/SKILL.md",
        &skill_fixture("browser-tools", "Browser tools"),
    );
    env.write(
        "agent/skills/pi-skills/browser-tools/EFFICIENCY.md",
        "No frontmatter here",
    );

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    let skills = loader.get_skills();
    assert!(skills.skills.iter().any(|s| s.name == "browser-tools"));
    assert!(!skills.diagnostics.iter().any(|d| {
        d.path
            .as_deref()
            .is_some_and(|p| p.ends_with("EFFICIENCY.md"))
    }));
}

#[tokio::test]
async fn discovers_prompts_from_agent_dir() {
    let env = LoaderEnv::new();
    env.write(
        "agent/prompts/test-prompt.md",
        "---\ndescription: A test prompt\n---\nPrompt content.",
    );

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    let prompts = loader.get_prompts();
    assert!(prompts.prompts.iter().any(|p| p.name == "test-prompt"));
}

#[tokio::test]
async fn prefers_project_resources_over_user_on_name_collisions() {
    let env = LoaderEnv::new();
    env.write("agent/prompts/commit.md", "User prompt");
    env.write("project/.pi/prompts/commit.md", "Project prompt");

    env.write(
        "agent/skills/collision-skill/SKILL.md",
        &skill_fixture("collision-skill", "user"),
    );
    env.write(
        "project/.pi/skills/collision-skill/SKILL.md",
        &skill_fixture("collision-skill", "project"),
    );

    env.write(
        "agent/themes/collision.json",
        "{\"name\": \"collision-theme\"}",
    );
    env.write(
        "project/.pi/themes/collision.json",
        "{\"name\": \"collision-theme\"}",
    );

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    let prompt = loader
        .get_prompts()
        .prompts
        .into_iter()
        .find(|p| p.name == "commit")
        .expect("project prompt");
    assert_eq!(
        PathBuf::from(&prompt.file_path),
        Path::new(&env.cwd).join(".pi/prompts/commit.md")
    );

    let skill = loader
        .get_skills()
        .skills
        .into_iter()
        .find(|s| s.name == "collision-skill")
        .expect("project skill");
    assert_eq!(
        PathBuf::from(&skill.file_path),
        Path::new(&env.cwd).join(".pi/skills/collision-skill/SKILL.md")
    );

    let theme = loader
        .get_themes()
        .themes
        .into_iter()
        .find(|t| t.name.as_deref() == Some("collision-theme"))
        .expect("project theme");
    assert_eq!(
        PathBuf::from(theme.source_path.expect("source path")),
        Path::new(&env.cwd).join(".pi/themes/collision.json")
    );
}

#[tokio::test]
async fn discovers_agents_md_context_files() {
    let env = LoaderEnv::new();
    env.write("project/AGENTS.md", "# Project Guidelines\n\nBe helpful.");

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    let agents_files = loader.get_agents_files();
    assert!(agents_files.iter().any(|f| f.path.contains("AGENTS.md")));
}

#[tokio::test]
async fn prefers_agents_override_md_within_each_directory_while_preserving_ancestor_layering() {
    let env = LoaderEnv::new();
    let nested_cwd = Path::new(&env.cwd).join("service");
    std::fs::create_dir_all(&nested_cwd).expect("nested cwd");
    env.write("agent/AGENTS.md", "global instructions");
    env.write("agent/AGENTS.override.md", "global override");
    env.write("project/AGENTS.md", "project instructions");
    std::fs::write(nested_cwd.join("AGENTS.md"), "service instructions").expect("write");
    std::fs::write(nested_cwd.join("AGENTS.override.md"), "service override").expect("write");

    let mut options = env.options();
    options.cwd = nested_cwd.to_string_lossy().into_owned();
    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    let read = |path: &str| std::fs::read_to_string(path).expect("read");
    let agents_files = loader.get_agents_files();
    assert_eq!(agents_files.len(), 3);
    assert_eq!(
        agents_files[0].path,
        Path::new(&env.agent_dir)
            .join("AGENTS.override.md")
            .to_string_lossy()
            .into_owned()
    );
    assert_eq!(agents_files[0].content, "global override");
    assert_eq!(
        agents_files[1].path,
        Path::new(&env.cwd)
            .join("AGENTS.md")
            .to_string_lossy()
            .into_owned()
    );
    assert_eq!(agents_files[1].content, "project instructions");
    assert_eq!(
        agents_files[2].path,
        nested_cwd
            .join("AGENTS.override.md")
            .to_string_lossy()
            .into_owned()
    );
    assert_eq!(agents_files[2].content, "service override");
    let _ = read;
}

#[tokio::test]
async fn ignores_context_file_candidates_that_are_directories() {
    let env = LoaderEnv::new();
    std::fs::create_dir_all(Path::new(&env.cwd).join("AGENTS.override.md")).expect("dir candidate");
    std::fs::create_dir_all(Path::new(&env.cwd).join("AGENTS.md")).expect("dir candidate");
    env.write("project/CLAUDE.md", "Fallback instructions");

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    let agents_files = loader.get_agents_files();
    assert!(
        agents_files.contains(&ContextFile {
            path: Path::new(&env.cwd)
                .join("CLAUDE.md")
                .to_string_lossy()
                .into_owned(),
            content: "Fallback instructions".to_string(),
        })
    );
}

#[tokio::test]
async fn skips_context_file_discovery_when_no_context_files_is_true() {
    let env = LoaderEnv::new();
    env.write(
        "project/AGENTS.override.md",
        "# Override Guidelines\n\nBe helpful.",
    );
    env.write("project/AGENTS.md", "# Project Guidelines\n\nBe helpful.");
    env.write("project/CLAUDE.md", "# Claude Guidelines\n\nBe helpful.");

    let mut options = env.options();
    options.no_context_files = true;
    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    assert_eq!(loader.get_agents_files(), Vec::<ContextFile>::new());
}

#[tokio::test]
async fn discovers_system_md_from_cwd_pi() {
    let env = LoaderEnv::new();
    env.write("project/.pi/SYSTEM.md", "You are a helpful assistant.");

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_system_prompt().as_deref(),
        Some("You are a helpful assistant.")
    );
}

#[tokio::test]
async fn skips_project_resources_that_require_trust_when_project_is_not_trusted() {
    let env = LoaderEnv::new();
    env.write("project/.pi/SYSTEM.md", "Project system prompt.");
    env.write("agent/SYSTEM.md", "Global system prompt.");
    env.write("agent/AGENTS.md", "Global instructions");
    env.write("project/AGENTS.md", "Project instructions");
    env.write(
        "project/.pi/extensions/project.ts",
        "throw new Error(\"should not load\");",
    );
    env.write(
        "project/.pi/skills/project-skill/SKILL.md",
        &skill_fixture("project-skill", "Project skill"),
    );
    env.write("project/.pi/prompts/project.md", "Project prompt");
    env.write(
        "project/.pi/themes/project.json",
        "{\"name\": \"project-theme\"}",
    );

    let mut loader = DefaultResourceLoader::new(env.options(), untrusted_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_system_prompt().as_deref(),
        Some("Global system prompt.")
    );
    let agents_files = loader.get_agents_files();
    assert!(agents_files.iter().any(|file| {
        file.path
            == Path::new(&env.agent_dir)
                .join("AGENTS.md")
                .to_string_lossy()
    }));
    assert!(
        agents_files
            .iter()
            .any(|file| file.path == Path::new(&env.cwd).join("AGENTS.md").to_string_lossy())
    );
    assert_eq!(loader.get_extensions().extensions.len(), 0);
    assert_eq!(loader.get_extensions().errors.len(), 0);
    assert!(
        !loader
            .get_skills()
            .skills
            .iter()
            .any(|skill| skill.name == "project-skill")
    );
    assert!(
        !loader
            .get_prompts()
            .prompts
            .iter()
            .any(|prompt| prompt.name == "project")
    );
    assert!(
        !loader
            .get_themes()
            .themes
            .iter()
            .any(|theme| theme.name.as_deref() == Some("project-theme"))
    );
}

#[tokio::test]
async fn discovers_append_system_md() {
    let env = LoaderEnv::new();
    env.write("project/.pi/APPEND_SYSTEM.md", "Additional instructions.");

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    assert!(
        loader
            .get_append_system_prompt()
            .iter()
            .any(|prompt| prompt.contains("Additional instructions."))
    );
}

// === system prompt sources ==================================================

#[tokio::test]
async fn exposes_discovered_project_system_md_as_the_system_prompt_source() {
    let env = LoaderEnv::new();
    let system_prompt_path = Path::new(&env.cwd).join(".pi/SYSTEM.md");
    env.write("project/.pi/SYSTEM.md", "Project system prompt.");

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_system_prompt().as_deref(),
        Some("Project system prompt.")
    );
    assert_eq!(
        loader.get_system_prompt_source(),
        Some(PromptSource {
            path: system_prompt_path.to_string_lossy().into_owned(),
        })
    );
}

#[tokio::test]
async fn exposes_discovered_global_system_md_as_the_system_prompt_source() {
    let env = LoaderEnv::new();
    let system_prompt_path = Path::new(&env.agent_dir).join("SYSTEM.md");
    env.write("agent/SYSTEM.md", "Global system prompt.");

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_system_prompt().as_deref(),
        Some("Global system prompt.")
    );
    assert_eq!(
        loader.get_system_prompt_source(),
        Some(PromptSource {
            path: system_prompt_path.to_string_lossy().into_owned(),
        })
    );
}

#[tokio::test]
async fn does_not_expose_literal_system_prompt_text_as_a_source() {
    let env = LoaderEnv::new();
    let mut options = env.options();
    options.system_prompt = Some("Literal system prompt.".to_string());

    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_system_prompt().as_deref(),
        Some("Literal system prompt.")
    );
    assert!(loader.get_system_prompt_source().is_none());
}

#[tokio::test]
async fn exposes_file_backed_system_prompt_options_as_a_source() {
    let env = LoaderEnv::new();
    let system_prompt_path = env.root.join("custom-system.md");
    std::fs::write(&system_prompt_path, "Custom system prompt.").expect("write");

    let mut options = env.options();
    options.system_prompt = Some(system_prompt_path.to_string_lossy().into_owned());

    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_system_prompt().as_deref(),
        Some("Custom system prompt.")
    );
    assert_eq!(
        loader.get_system_prompt_source(),
        Some(PromptSource {
            path: system_prompt_path.to_string_lossy().into_owned(),
        })
    );
}

#[tokio::test]
async fn exposes_discovered_append_system_md_as_an_append_system_prompt_source() {
    let env = LoaderEnv::new();
    let append_system_prompt_path = Path::new(&env.cwd).join(".pi/APPEND_SYSTEM.md");
    env.write("project/.pi/APPEND_SYSTEM.md", "Project append prompt.");

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_append_system_prompt(),
        vec!["Project append prompt.".to_string()]
    );
    assert_eq!(
        loader.get_append_system_prompt_sources(),
        vec![PromptSource {
            path: append_system_prompt_path.to_string_lossy().into_owned(),
        }]
    );
}

#[tokio::test]
async fn does_not_expose_literal_append_system_prompt_text_as_a_source() {
    let env = LoaderEnv::new();
    let mut options = env.options();
    options.append_system_prompt = Some(vec!["Literal append prompt.".to_string()]);

    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_append_system_prompt(),
        vec!["Literal append prompt.".to_string()]
    );
    assert_eq!(
        loader.get_append_system_prompt_sources(),
        Vec::<PromptSource>::new()
    );
}

#[tokio::test]
async fn only_exposes_file_backed_append_system_prompt_options_as_sources() {
    let env = LoaderEnv::new();
    let append_system_prompt_path = env.root.join("custom-append.md");
    std::fs::write(&append_system_prompt_path, "Custom append prompt.").expect("write");

    let mut options = env.options();
    options.append_system_prompt = Some(vec![
        append_system_prompt_path.to_string_lossy().into_owned(),
        "Literal append prompt.".to_string(),
    ]);

    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_append_system_prompt(),
        vec![
            "Custom append prompt.".to_string(),
            "Literal append prompt.".to_string()
        ]
    );
    assert_eq!(
        loader.get_append_system_prompt_sources(),
        vec![PromptSource {
            path: append_system_prompt_path.to_string_lossy().into_owned(),
        }]
    );
}

// === extendResources ========================================================

#[tokio::test]
async fn loads_skills_and_prompts_with_extension_metadata() {
    let env = LoaderEnv::new();
    let extra_skill_dir = env.root.join("extra-skills/extra-skill");
    let skill_path = extra_skill_dir.join("SKILL.md");
    std::fs::create_dir_all(extra_skill_dir.as_path()).expect("mkdir");
    std::fs::write(
        skill_path.as_path(),
        skill_fixture("extra-skill", "Extra skill"),
    )
    .expect("write");

    let extra_prompt_dir = env.root.join("extra-prompts");
    let prompt_path = extra_prompt_dir.join("extra.md");
    std::fs::create_dir_all(&extra_prompt_dir).expect("mkdir");
    std::fs::write(
        prompt_path.as_path(),
        "---\ndescription: Extra prompt\n---\nExtra prompt content",
    )
    .expect("write");

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    loader.extend_resources(&ResourceExtensionPaths {
        skill_paths: vec![PathWithMetadata {
            path: extra_skill_dir.to_string_lossy().into_owned(),
            metadata: PathMetadata {
                source: "extension:extra".to_string(),
                scope: SourceScope::Temporary,
                origin: SourceOrigin::TopLevel,
                base_dir: Some(extra_skill_dir.to_string_lossy().into_owned()),
            },
        }],
        prompt_paths: vec![PathWithMetadata {
            path: prompt_path.to_string_lossy().into_owned(),
            metadata: PathMetadata {
                source: "extension:extra".to_string(),
                scope: SourceScope::Temporary,
                origin: SourceOrigin::TopLevel,
                base_dir: Some(extra_prompt_dir.to_string_lossy().into_owned()),
            },
        }],
        theme_paths: Vec::new(),
    });

    let loaded_skill = loader
        .get_skills()
        .skills
        .into_iter()
        .find(|skill| skill.name == "extra-skill")
        .expect("extra skill");
    assert_eq!(loaded_skill.source_info.source, "extension:extra");
    assert_eq!(PathBuf::from(&loaded_skill.source_info.path), skill_path);

    let loaded_prompt = loader
        .get_prompts()
        .prompts
        .into_iter()
        .find(|prompt| prompt.name == "extra")
        .expect("extra prompt");
    assert_eq!(loaded_prompt.source_info.source, "extension:extra");
    assert_eq!(PathBuf::from(&loaded_prompt.source_info.path), prompt_path);
}

#[tokio::test]
async fn loads_extension_resources_returned_as_file_urls() {
    let env = LoaderEnv::new();
    let extra_skill_dir = env.root.join("extra skills/file-url-skill");
    let skill_path = extra_skill_dir.join("SKILL.md");
    std::fs::create_dir_all(extra_skill_dir.as_path()).expect("mkdir");
    std::fs::write(
        skill_path.as_path(),
        skill_fixture("file-url-skill", "File URL skill"),
    )
    .expect("write");

    let file_url = url::Url::from_file_path(extra_skill_dir.as_path())
        .expect("file url")
        .to_string();

    let mut loader = DefaultResourceLoader::new(env.options(), file_manager(&env));
    loader.reload(None).await;

    loader.extend_resources(&ResourceExtensionPaths {
        skill_paths: vec![PathWithMetadata {
            path: file_url,
            metadata: PathMetadata {
                source: "extension:file-url".to_string(),
                scope: SourceScope::Temporary,
                origin: SourceOrigin::TopLevel,
                base_dir: Some(extra_skill_dir.to_string_lossy().into_owned()),
            },
        }],
        prompt_paths: Vec::new(),
        theme_paths: Vec::new(),
    });

    let skills = loader.get_skills();
    assert_eq!(skills.diagnostics, Vec::new());
    let loaded_skill = skills
        .skills
        .into_iter()
        .find(|skill| skill.name == "file-url-skill")
        .expect("file-url skill");
    assert_eq!(PathBuf::from(&loaded_skill.file_path), skill_path);
    assert_eq!(loaded_skill.source_info.source, "extension:file-url");
}

// === noSkills option ========================================================

#[tokio::test]
async fn skips_skill_discovery_when_no_skills_is_true() {
    let env = LoaderEnv::new();
    env.write(
        "agent/skills/test-skill.md",
        &skill_fixture("test-skill", "A test skill"),
    );

    let mut options = env.options();
    options.no_skills = true;
    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    assert_eq!(loader.get_skills().skills, Vec::<Skill>::new());
}

#[tokio::test]
async fn still_loads_additional_skill_paths_when_no_skills_is_true() {
    let env = LoaderEnv::new();
    let custom_skill_dir = env.root.join("custom-skills");
    std::fs::create_dir_all(custom_skill_dir.as_path()).expect("mkdir");
    std::fs::write(
        custom_skill_dir.join("custom.md"),
        skill_fixture("custom", "Custom skill"),
    )
    .expect("write");

    let mut options = env.options();
    options.no_skills = true;
    options.additional_skill_paths = vec![custom_skill_dir.to_string_lossy().into_owned()];
    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    let skills = loader.get_skills();
    assert!(skills.skills.iter().any(|s| s.name == "custom"));
}

// === override functions =====================================================

#[tokio::test]
async fn applies_skills_override() {
    let env = LoaderEnv::new();
    let injected = Skill {
        name: "injected".to_string(),
        description: "Injected skill".to_string(),
        file_path: "/fake/path".to_string(),
        base_dir: "/fake".to_string(),
        source_info: pi_coding_agent::source_info::create_synthetic_source_info(
            "/fake/path",
            &pi_coding_agent::source_info::SyntheticSourceOptions {
                source: "custom".to_string(),
                ..pi_coding_agent::source_info::SyntheticSourceOptions::default()
            },
        ),
        disable_model_invocation: false,
    };

    let mut options = env.options();
    options.skills_override = Some(Box::new(move |_base| {
        pi_coding_agent::resource_loader::LoadedSkillsResult {
            skills: vec![injected.clone()],
            diagnostics: Vec::new(),
        }
    }));

    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    let skills = loader.get_skills();
    assert_eq!(skills.skills.len(), 1);
    assert_eq!(skills.skills[0].name, "injected");
}

#[tokio::test]
async fn applies_system_prompt_override() {
    let env = LoaderEnv::new();
    let mut options = env.options();
    options.system_prompt_override =
        Some(Box::new(|_base| Some("Custom system prompt".to_string())));

    let mut loader = DefaultResourceLoader::new(options, file_manager(&env));
    loader.reload(None).await;

    assert_eq!(
        loader.get_system_prompt().as_deref(),
        Some("Custom system prompt")
    );
}

// === settings-driven resource entries =======================================

#[tokio::test]
async fn settings_override_patterns_disable_auto_discovered_resources() {
    let env = LoaderEnv::new();
    env.write(
        "agent/skills/skip-skill/SKILL.md",
        &skill_fixture("skip-skill", "Skip me"),
    );
    env.write("agent/prompts/skip.md", "Skip prompt");
    env.write("agent/themes/skip.json", "{}");

    let settings = Settings::new();
    let mut manager =
        SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
    manager.set_extension_paths(&["-extensions/disabled.ts".to_string()]);
    manager.set_skill_paths(&["-skills/skip-skill".to_string()]);
    manager.set_prompt_template_paths(&["-prompts/skip.md".to_string()]);
    manager.set_theme_paths(&["-themes/skip.json".to_string()]);

    let mut loader = DefaultResourceLoader::new(env.options(), manager);
    loader.reload(None).await;

    assert!(
        !loader
            .get_skills()
            .skills
            .iter()
            .any(|s| s.name == "skip-skill")
    );
    assert!(
        !loader
            .get_prompts()
            .prompts
            .iter()
            .any(|p| p.name == "skip")
    );
    assert!(!loader.get_themes().themes.iter().any(|t| {
        t.source_path
            .as_deref()
            .is_some_and(|p| p.ends_with("skip.json"))
    }));
}

#[tokio::test]
async fn settings_entries_load_with_user_scope_metadata() {
    let env = LoaderEnv::new();
    env.write(
        "agent/skills/test-skill.md",
        &skill_fixture("test-skill", "A test skill"),
    );

    let settings = Settings::new();
    let mut manager =
        SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
    manager.set_skill_paths(&["skills/test-skill.md".to_string()]);

    let mut loader = DefaultResourceLoader::new(env.options(), manager);
    loader.reload(None).await;

    let skills = loader.get_skills();
    assert_eq!(skills.skills.len(), 1);
    assert_eq!(skills.skills[0].source_info.source, "local");
    assert!(matches!(
        skills.skills[0].source_info.scope,
        SourceScope::User
    ));
    assert!(matches!(
        skills.skills[0].source_info.origin,
        SourceOrigin::TopLevel
    ));
}

// === loadProjectContextFiles - nested worktree dedup ========================

/// Builds a linked-worktree skeleton (no git binary needed): the main
/// repo's `.git/worktrees/<name>/` holds `HEAD` plus a `commondir` pointing
/// back at the main `.git`, and the worktree's working tree carries a
/// `.git` *file* whose `gitdir:` resolves to it.
fn link_worktree(main_dir: &Path, worktree_dir: &Path, name: &str) {
    let git_dir = main_dir.join(".git").join("worktrees").join(name);
    std::fs::create_dir_all(&git_dir).expect("git dir");
    // The main repo's own `.git` is a real git dir with a HEAD, as git
    // writes it.
    std::fs::write(main_dir.join(".git").join("HEAD"), "ref: refs/heads/main\n").expect("write");
    std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/feat\n").expect("write");
    // commondir is relative to the worktree gitdir and points at the main
    // .git.
    std::fs::write(git_dir.join("commondir"), "../..").expect("write");
    std::fs::write(
        worktree_dir.join(".git"),
        format!("gitdir: {}\n", git_dir.to_string_lossy()),
    )
    .expect("write");
}

/// Main repo at `<root>/outer/main` with a linked worktree at
/// `main/worktrees/feat`. Each case writes only the AGENTS.md files it
/// needs.
fn setup_nested_worktree(root: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let outer = root.join("outer");
    let main = outer.join("main");
    let worktree = main.join("worktrees").join("feat");
    let worktree_src = worktree.join("src");
    std::fs::create_dir_all(&worktree_src).expect("worktree src");
    link_worktree(&main, &worktree, "feat");
    (outer, main, worktree, worktree_src)
}

fn agent_dir_of(root: &Path) -> String {
    root.join("agent").to_string_lossy().into_owned()
}

fn contents(files: &[ContextFile]) -> Vec<String> {
    files.iter().map(|file| file.content.clone()).collect()
}

#[test]
fn skips_the_main_repos_duplicate_when_the_worktree_root_has_its_own_context() {
    let env = LoaderEnv::new();
    let (_outer, main, _worktree, worktree_src) = setup_nested_worktree(&env.root);
    std::fs::write(main.join("AGENTS.md"), "main repo instructions").expect("write");
    std::fs::write(
        worktree_src
            .parent()
            .expect("worktree root")
            .join("AGENTS.md"),
        "worktree instructions",
    )
    .expect("write");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &worktree_src.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    assert_eq!(contents(&files), vec!["worktree instructions".to_string()]);
}

#[test]
fn still_inherits_the_main_repos_context_when_the_worktree_root_has_none() {
    let env = LoaderEnv::new();
    let (_outer, main, _worktree, worktree_src) = setup_nested_worktree(&env.root);
    std::fs::write(main.join("AGENTS.md"), "main repo instructions").expect("write");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &worktree_src.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    assert_eq!(contents(&files), vec!["main repo instructions".to_string()]);
}

#[test]
fn only_skips_the_same_filename_not_a_differently_named_context_file() {
    // The repo tracks CLAUDE.md; the worktree adds an AGENTS.md, which
    // loadContextFileFromDir prefers. The main repo's CLAUDE.md is nobody's
    // duplicate, so dropping it would lose its content entirely.
    let env = LoaderEnv::new();
    let (_outer, main, worktree, worktree_src) = setup_nested_worktree(&env.root);
    std::fs::write(main.join("CLAUDE.md"), "main repo instructions").expect("write");
    std::fs::write(worktree.join("AGENTS.md"), "worktree instructions").expect("write");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &worktree_src.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    assert_eq!(
        contents(&files),
        vec![
            "main repo instructions".to_string(),
            "worktree instructions".to_string()
        ]
    );
}

#[test]
fn does_not_skip_the_containers_context_in_a_bare_layout() {
    // `git clone --bare proj/.bare` + `git worktree add ../main` makes
    // commondir `../..`, so dirname(commonGitDir) is `proj` - a plain
    // directory that tracks nothing. Its AGENTS.md is not a duplicate of
    // the worktree's. Layout below matches what real git writes for this
    // setup.
    let env = LoaderEnv::new();
    let proj = env.root.join("proj");
    let bare = proj.join(".bare");
    let worktree = proj.join("main");
    let worktree_git_dir = bare.join("worktrees").join("main");
    std::fs::create_dir_all(&worktree_git_dir).expect("git dir");
    std::fs::create_dir_all(&worktree).expect("worktree");
    std::fs::write(bare.join("HEAD"), "ref: refs/heads/main\n").expect("write");
    std::fs::write(worktree_git_dir.join("HEAD"), "ref: refs/heads/main\n").expect("write");
    std::fs::write(worktree_git_dir.join("commondir"), "../..").expect("write");
    std::fs::write(
        worktree.join(".git"),
        format!("gitdir: {}\n", worktree_git_dir.to_string_lossy()),
    )
    .expect("write");
    std::fs::write(proj.join("AGENTS.md"), "container instructions").expect("write");
    std::fs::write(worktree.join("AGENTS.md"), "worktree instructions").expect("write");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &worktree.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    assert_eq!(
        contents(&files),
        vec![
            "container instructions".to_string(),
            "worktree instructions".to_string()
        ]
    );
}

#[test]
fn keeps_loading_ancestors_above_the_main_repo() {
    let env = LoaderEnv::new();
    let (outer, main, worktree, worktree_src) = setup_nested_worktree(&env.root);
    std::fs::write(outer.join("AGENTS.md"), "outer instructions").expect("write");
    std::fs::write(main.join("AGENTS.md"), "main repo instructions").expect("write");
    std::fs::write(worktree.join("AGENTS.md"), "worktree instructions").expect("write");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &worktree_src.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    // Only the main repo root's duplicate is dropped; the unrelated dir
    // above it stays.
    assert_eq!(
        contents(&files),
        vec![
            "outer instructions".to_string(),
            "worktree instructions".to_string()
        ]
    );
}

#[test]
fn does_not_skip_anything_for_a_sibling_worktree() {
    // git worktree add ../feat puts the worktree beside the main repo, so
    // no duplicate is ever encountered and ancestors above it are
    // unrelated.
    let env = LoaderEnv::new();
    let outer = env.root.join("outer");
    let main = outer.join("main");
    let sib = outer.join("sib-feat");
    let sib_src = sib.join("src");
    std::fs::create_dir_all(&sib_src).expect("sib src");
    std::fs::create_dir_all(&main).expect("main");
    std::fs::write(outer.join("AGENTS.md"), "outer instructions").expect("write");
    std::fs::write(sib.join("AGENTS.md"), "sibling worktree instructions").expect("write");
    link_worktree(&main, &sib, "sib");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &sib_src.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    assert_eq!(
        contents(&files),
        vec![
            "outer instructions".to_string(),
            "sibling worktree instructions".to_string()
        ]
    );
}

#[test]
fn does_not_skip_the_superprojects_context_from_inside_a_submodule() {
    // A submodule's `.git` file is also `gitdir:`-style, but its gitdir has
    // no commondir, so it resolves under `.git/modules` - never an ancestor
    // of cwd.
    let env = LoaderEnv::new();
    let sup = env.root.join("super");
    let sub = sup.join("vendor").join("lib");
    let sub_src = sub.join("src");
    std::fs::create_dir_all(&sub_src).expect("sub src");
    std::fs::write(sup.join("AGENTS.md"), "superproject instructions").expect("write");
    std::fs::write(sub.join("AGENTS.md"), "submodule instructions").expect("write");
    let sub_git_dir = sup.join(".git").join("modules").join("vendor").join("lib");
    std::fs::create_dir_all(&sub_git_dir).expect("git dir");
    std::fs::write(sub_git_dir.join("HEAD"), "ref: refs/heads/main\n").expect("write");
    std::fs::write(
        sub.join(".git"),
        format!("gitdir: {}\n", sub_git_dir.to_string_lossy()),
    )
    .expect("write");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &sub_src.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    assert_eq!(
        contents(&files),
        vec![
            "superproject instructions".to_string(),
            "submodule instructions".to_string()
        ]
    );
}

#[test]
fn keeps_climbing_past_an_ordinary_repo_root() {
    let env = LoaderEnv::new();
    let outer = env.root.join("outer");
    let repo = outer.join("repo");
    let leaf = repo.join("src");
    std::fs::create_dir_all(&leaf).expect("leaf");
    std::fs::create_dir_all(repo.join(".git")).expect("git dir");
    std::fs::write(repo.join(".git").join("HEAD"), "ref: refs/heads/main\n").expect("write");
    std::fs::write(outer.join("AGENTS.md"), "outer instructions").expect("write");
    std::fs::write(repo.join("AGENTS.md"), "repo instructions").expect("write");
    std::fs::write(leaf.join("AGENTS.md"), "leaf instructions").expect("write");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &leaf.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    assert_eq!(
        contents(&files),
        vec![
            "outer instructions".to_string(),
            "repo instructions".to_string(),
            "leaf instructions".to_string()
        ]
    );
}

#[test]
fn climbs_normally_when_the_gitdir_target_does_not_exist() {
    let env = LoaderEnv::new();
    let repo = env.root.join("corrupt");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).expect("src");
    std::fs::write(
        repo.join(".git"),
        "gitdir: /nonexistent/path/worktrees/feat\n",
    )
    .expect("write");
    std::fs::write(repo.join("AGENTS.md"), "repo instructions").expect("write");
    std::fs::write(src.join("AGENTS.md"), "src instructions").expect("write");

    let files = pi_coding_agent::resource_loader::load_project_context_files(
        &src.to_string_lossy(),
        &agent_dir_of(&env.root),
    );

    assert_eq!(
        contents(&files),
        vec![
            "repo instructions".to_string(),
            "src instructions".to_string()
        ]
    );
}

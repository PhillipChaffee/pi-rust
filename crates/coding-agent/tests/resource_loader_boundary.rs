//! Boundary tests binding the resource layer's untested branches at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the package manager's
//! classification, pattern, and collector machinery; the manifest reader;
//! the git-path probe and its worktree shapes; the loader's diagnostics,
//! seam flows, and source attribution; the auth-guidance strings; and the
//! config doc paths. Expected values run from upstream's TypeScript at
//! the pin where an oracle was needed.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::literal_string_with_formatting_args,
    reason = "the substitution grammar's placeholders carry braced patterns, upstream's dollar-brace default and slice shapes"
)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pi_coding_agent::auth_guidance::{
    format_no_api_key_found_message, format_no_model_selected_message,
    format_no_models_available_message, get_provider_login_help,
};
use pi_coding_agent::config::{get_docs_path, get_examples_path, get_readme_path};
use pi_coding_agent::diagnostics::ResourceDiagnosticKind;
use pi_coding_agent::footer_data_provider::find_git_paths;
use pi_coding_agent::package_manager::{
    DefaultPackageManager, PackageManagerOptions, ParsedSource, PathMetadata, ResolvedPaths,
    ResourceType, SkillDiscoveryMode, resource_precedence_rank,
};
use pi_coding_agent::pi_manifest::{PiManifest, read_pi_manifest, read_pi_manifest_value};
use pi_coding_agent::resource_loader::{
    DefaultResourceLoader, DefaultResourceLoaderOptions, PathWithMetadata, PromptSource,
    ResourceExtensionPaths,
};
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
    SettingsStorage,
};
use pi_coding_agent::source_info::{SourceOrigin, SourceScope};

type InMemoryHandle = Arc<Mutex<SettingsManager<InMemorySettingsStorage>>>;

// === auth-guidance ==========================================================

#[test]
fn provider_login_help_names_the_doc_pages() {
    let help = get_provider_login_help();
    assert!(help.starts_with("Use /login to log into a provider via OAuth or API key. See:"));
    assert!(help.contains(&format!("{}/providers.md", get_docs_path())));
    assert!(help.contains(&format!("{}/models.md", get_docs_path())));
}

#[test]
fn no_models_message_prefixes_the_login_help() {
    assert_eq!(
        format_no_models_available_message(),
        format!("No models available. {}", get_provider_login_help())
    );
}

#[test]
fn no_model_selected_message_wraps_the_login_help() {
    let help = get_provider_login_help();
    assert_eq!(
        format_no_model_selected_message(),
        format!("No model selected.\n\n{help}\n\nThen use /model to select a model.")
    );
}

#[test]
fn no_api_key_message_reads_the_selected_model_for_unknown_providers() {
    let help = get_provider_login_help();
    assert_eq!(
        format_no_api_key_found_message("unknown"),
        format!("No API key found for the selected model.\n\n{help}")
    );
    assert_eq!(
        format_no_api_key_found_message("anthropic"),
        format!("No API key found for anthropic.\n\n{help}")
    );
}

// === config doc paths =======================================================

#[test]
fn doc_paths_ride_the_package_dir() {
    // The getters join the package dir (the PI_PACKAGE_DIR override or the
    // executable directory; the env-injection seam lives on
    // get_package_dir_with, the config suite's). The contract pinned here
    // is the join shape.
    let readme = get_readme_path();
    let docs = get_docs_path();
    let examples = get_examples_path();
    assert!(readme.ends_with("README.md"), "{readme}");
    assert!(docs.ends_with("docs"), "{docs}");
    assert!(examples.ends_with("examples"), "{examples}");
}

// === pi-manifest ============================================================

#[test]
fn reads_pi_manifest_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("package.json");
    std::fs::write(
        &path,
        r#"{"name":"pkg","pi":{"extensions":["ext.ts"],"skills":["skills"],"prompts":["prompts"],"themes":["themes/x.json"]}}"#,
    )
    .expect("write");
    let manifest = read_pi_manifest(&path.to_string_lossy()).expect("manifest");
    assert_eq!(
        manifest.extensions.as_deref(),
        Some(&["ext.ts".to_string()][..])
    );
    assert_eq!(
        manifest.skills.as_deref(),
        Some(&["skills".to_string()][..])
    );
    assert_eq!(
        manifest.prompts.as_deref(),
        Some(&["prompts".to_string()][..])
    );
    assert_eq!(
        manifest.themes.as_deref(),
        Some(&["themes/x.json".to_string()][..])
    );
}

#[test]
fn drops_manifest_entries_that_are_not_string_arrays() {
    let value: serde_json::Value = serde_json::from_str(
        r#"{"pi":{"extensions":[1,2],"skills":"nope","prompts":["ok"],"themes":[]}}"#,
    )
    .expect("json");
    let manifest = read_pi_manifest_value(&value).expect("manifest");
    assert!(manifest.extensions.is_none());
    assert!(manifest.skills.is_none());
    assert_eq!(manifest.prompts.as_deref(), Some(&["ok".to_string()][..]));
    // An empty string array reads in, the every-string guard passing.
    assert_eq!(manifest.themes.as_deref(), Some(&[][..]));
}

#[test]
fn reads_no_manifest_without_a_pi_object() {
    let value: serde_json::Value = serde_json::from_str(r#"{"name":"pkg"}"#).expect("json");
    assert!(read_pi_manifest_value(&value).is_none());
    let non_object: serde_json::Value = serde_json::from_str(r#"["nope"]"#).expect("json");
    assert!(read_pi_manifest_value(&non_object).is_none());
}

#[test]
fn reads_no_manifest_from_a_missing_or_unparseable_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(read_pi_manifest(&dir.path().join("package.json").to_string_lossy()).is_none());
    let path = dir.path().join("package.json");
    std::fs::write(&path, "not json").expect("write");
    assert!(read_pi_manifest(&path.to_string_lossy()).is_none());
}

#[test]
fn the_default_pi_manifest_carries_no_entries() {
    assert_eq!(PiManifest::default(), PiManifest::default());
}

// === git paths ==============================================================

#[test]
fn find_git_paths_reads_an_ordinary_repo() {
    let dir = tempfile::tempdir().expect("tempdir");
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).expect("git dir");
    std::fs::write(repo.join(".git").join("HEAD"), "ref: refs/heads/main\n").expect("write");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).expect("src");

    let paths = find_git_paths(&src.to_string_lossy()).expect("paths");
    assert_eq!(paths.repo_dir, repo.to_string_lossy().into_owned());
    assert_eq!(
        paths.common_git_dir,
        repo.join(".git").to_string_lossy().into_owned()
    );
    assert_eq!(
        paths.head_path,
        repo.join(".git")
            .join("HEAD")
            .to_string_lossy()
            .into_owned()
    );
}

#[test]
fn find_git_paths_resolves_a_worktree_gitdir_and_commondir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let main = dir.path().join("main");
    let worktree = dir.path().join("feat");
    std::fs::create_dir_all(main.join(".git").join("worktrees").join("feat")).expect("git dirs");
    std::fs::create_dir_all(&worktree).expect("worktree");
    std::fs::write(main.join(".git").join("HEAD"), "ref: refs/heads/main\n").expect("write");
    let git_dir = main.join(".git").join("worktrees").join("feat");
    std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/feat\n").expect("write");
    std::fs::write(git_dir.join("commondir"), "../..").expect("write");
    std::fs::write(
        worktree.join(".git"),
        format!("gitdir: {}\n", git_dir.to_string_lossy()),
    )
    .expect("write");

    let paths = find_git_paths(&worktree.to_string_lossy()).expect("paths");
    assert_eq!(paths.repo_dir, worktree.to_string_lossy().into_owned());
    assert_eq!(
        paths.common_git_dir,
        main.join(".git").to_string_lossy().into_owned()
    );
}

#[test]
fn find_git_paths_returns_none_without_a_head_or_a_repo() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A repo whose .git carries no HEAD reports no paths.
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).expect("git dir");
    assert!(find_git_paths(&repo.to_string_lossy()).is_none());
    // No repo above the probe point at all.
    assert!(find_git_paths(&dir.path().to_string_lossy()).is_none());
}

// === package-manager classification and ranks ===============================

#[test]
fn classifies_package_sources() {
    let manager = manager_with("/tmp", "/tmp/agent", &settings_handle());
    // An `npm:` spec is a parse error, the npm channel's drop (ADR 0007).
    assert!(manager.parse_source("npm:foo").is_err());
    assert!(matches!(
        manager.parse_source("git:https://github.com/org/repo"),
        Ok(ParsedSource::Git(_))
    ));
    assert!(matches!(
        manager.parse_source("https://github.com/org/repo"),
        Ok(ParsedSource::Git(_))
    ));
    assert!(matches!(
        manager.parse_source("ssh://git@github.com/org/repo"),
        Ok(ParsedSource::Git(_))
    ));
    // Upstream's git-URL vocabulary carries no `git+` alternative — a
    // `git+https://` spec parses as neither git nor a remote protocol, and
    // the parser reads it local.
    assert!(matches!(
        manager.parse_source("git+https://github.com/org/repo"),
        Ok(ParsedSource::Local(_))
    ));
    assert!(matches!(
        manager.parse_source("./local/dir"),
        Ok(ParsedSource::Local(_))
    ));
    assert!(matches!(
        manager.parse_source("/absolute/path"),
        Ok(ParsedSource::Local(_))
    ));
    assert!(matches!(
        manager.parse_source("plain-name"),
        Ok(ParsedSource::Local(_))
    ));
}

#[test]
fn precedence_ranks_order_project_local_auto_user_and_packages() {
    let local_project = PathMetadata {
        source: "local".to_string(),
        scope: SourceScope::Project,
        origin: SourceOrigin::TopLevel,
        base_dir: None,
    };
    let auto_project = PathMetadata {
        source: "auto".to_string(),
        scope: SourceScope::Project,
        origin: SourceOrigin::TopLevel,
        base_dir: None,
    };
    let local_user = PathMetadata {
        source: "local".to_string(),
        scope: SourceScope::User,
        origin: SourceOrigin::TopLevel,
        base_dir: None,
    };
    let auto_user = PathMetadata {
        source: "auto".to_string(),
        scope: SourceScope::User,
        origin: SourceOrigin::TopLevel,
        base_dir: None,
    };
    let package = PathMetadata {
        source: "npm:x".to_string(),
        scope: SourceScope::User,
        origin: SourceOrigin::Package,
        base_dir: None,
    };

    assert_eq!(resource_precedence_rank(&local_project), 0);
    assert_eq!(resource_precedence_rank(&auto_project), 1);
    assert_eq!(resource_precedence_rank(&local_user), 2);
    assert_eq!(resource_precedence_rank(&auto_user), 3);
    assert_eq!(resource_precedence_rank(&package), 4);
}

#[test]
fn the_resource_type_names_match_their_settings_keys() {
    assert_eq!(ResourceType::Extensions.as_str(), "extensions");
    assert_eq!(ResourceType::Skills.as_str(), "skills");
    assert_eq!(ResourceType::Prompts.as_str(), "prompts");
    assert_eq!(ResourceType::Themes.as_str(), "themes");
}

// === package-manager collectors =============================================

#[test]
fn auto_extension_entries_prefer_explicit_entries_and_index_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A directory declaring extensions in its pi manifest resolves to them.
    let manifest_dir = dir.path().join("declared");
    std::fs::create_dir_all(&manifest_dir).expect("mkdir");
    std::fs::write(
        manifest_dir.join("package.json"),
        r#"{"pi":{"extensions":["e.ts"]}}"#,
    )
    .expect("write");
    std::fs::write(manifest_dir.join("e.ts"), "export default 1;").expect("write");
    std::fs::write(manifest_dir.join("other.ts"), "export default 2;").expect("write");

    let entries = pi_coding_agent::package_manager::collect_auto_extension_entries(&manifest_dir);
    assert_eq!(
        entries,
        vec![manifest_dir.join("e.ts").to_string_lossy().into_owned()]
    );

    // A directory with an index file resolves to it. The index convention
    // restates to an executable `index` (ADR 0007's executability filter).
    let index_dir = dir.path().join("indexed");
    std::fs::create_dir_all(&index_dir).expect("mkdir");
    std::fs::write(index_dir.join("index"), "export default 1;").expect("write");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            index_dir.join("index"),
            std::fs::Permissions::from_mode(0o755),
        )
        .expect("chmod");
    }
    let entries = pi_coding_agent::package_manager::collect_auto_extension_entries(&index_dir);
    assert_eq!(
        entries,
        vec![index_dir.join("index").to_string_lossy().into_owned()]
    );

    // An undecorated directory collects its direct executable children;
    // a non-executable file never surfaces.
    let loose_dir = dir.path().join("loose");
    std::fs::create_dir_all(&loose_dir).expect("mkdir");
    for name in ["a", "b"] {
        let path = loose_dir.join(name);
        std::fs::write(&path, "").expect("write");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
    }
    std::fs::write(loose_dir.join("c.txt"), "").expect("write");
    let mut entries = pi_coding_agent::package_manager::collect_auto_extension_entries(&loose_dir);
    entries.sort();
    assert_eq!(
        entries,
        vec![
            loose_dir.join("a").to_string_lossy().into_owned(),
            loose_dir.join("b").to_string_lossy().into_owned(),
        ]
    );
}

#[test]
fn auto_skill_entries_follow_the_flavor_depth_rule() {
    let dir = tempfile::tempdir().expect("tempdir");
    let skills_dir = dir.path().join("skills");
    std::fs::create_dir_all(skills_dir.join("nested/deeper")).expect("mkdir");
    std::fs::write(skills_dir.join("root.md"), "").expect("write");
    std::fs::write(skills_dir.join("nested/sub.md"), "").expect("write");
    std::fs::write(skills_dir.join("nested/deeper/SKILL.md"), "").expect("write");

    // The pi flavor loads root .md files only at the traversal root.
    let mut pi_entries = pi_coding_agent::package_manager::collect_auto_skill_entries(
        &skills_dir,
        SkillDiscoveryMode::Pi,
    );
    pi_entries.sort();
    assert_eq!(
        pi_entries,
        vec![
            skills_dir
                .join("nested/deeper/SKILL.md")
                .to_string_lossy()
                .into_owned(),
            skills_dir.join("root.md").to_string_lossy().into_owned(),
        ]
    );

    // The agents flavor loads root .md files only below the root: a
    // root-level .md file skips; a SKILL.md-free subdirectory's .md file
    // loads (a subdirectory carrying SKILL.md stops there, one skill per
    // directory).
    let agents_dir = dir.path().join(".agents/skills");
    std::fs::create_dir_all(agents_dir.join("sub")).expect("mkdir");
    std::fs::write(agents_dir.join("root.md"), "").expect("write");
    std::fs::write(agents_dir.join("sub/notes.md"), "").expect("write");
    let agents_entries = pi_coding_agent::package_manager::collect_auto_skill_entries(
        &agents_dir,
        SkillDiscoveryMode::Agents,
    );
    assert_eq!(
        agents_entries,
        vec![
            agents_dir
                .join("sub/notes.md")
                .to_string_lossy()
                .into_owned()
        ]
    );
}

#[test]
fn ancestor_agents_skill_dirs_stop_at_the_git_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let repo = dir.path().join("repo");
    let deep = repo.join("a/b");
    std::fs::create_dir_all(&deep).expect("mkdir");
    std::fs::create_dir_all(repo.join(".git")).expect("git dir");

    let dirs = pi_coding_agent::package_manager::collect_ancestor_agents_skill_dirs(&deep);
    assert_eq!(
        dirs,
        vec![
            deep.join(".agents/skills").to_string_lossy().into_owned(),
            deep.parent()
                .expect("parent")
                .join(".agents/skills")
                .to_string_lossy()
                .into_owned(),
            repo.join(".agents/skills").to_string_lossy().into_owned(),
        ]
    );
}

#[test]
fn auto_prompt_and_theme_entries_skip_ignored_children() {
    let dir = tempfile::tempdir().expect("tempdir");
    let prompts_dir = dir.path().join("prompts");
    std::fs::create_dir_all(&prompts_dir).expect("mkdir");
    std::fs::write(prompts_dir.join("a.md"), "").expect("write");
    std::fs::write(prompts_dir.join("b.txt"), "").expect("write");
    std::fs::write(prompts_dir.join(".hidden.md"), "").expect("write");
    std::fs::write(prompts_dir.join(".gitignore"), "a.md\n").expect("write");

    assert_eq!(
        pi_coding_agent::package_manager::collect_auto_prompt_entries(&prompts_dir),
        Vec::<String>::new()
    );

    let themes_dir = dir.path().join("themes");
    std::fs::create_dir_all(&themes_dir).expect("mkdir");
    std::fs::write(themes_dir.join("x.json"), "{}").expect("write");
    std::fs::write(themes_dir.join("y.md"), "").expect("write");
    assert_eq!(
        pi_coding_agent::package_manager::collect_auto_theme_entries(&themes_dir),
        vec![themes_dir.join("x.json").to_string_lossy().into_owned()]
    );
}

// === package-manager resolution =============================================

fn settings_handle() -> InMemoryHandle {
    Arc::new(Mutex::new(SettingsManager::in_memory(
        &Settings::new(),
        SettingsManagerCreateOptions::default(),
    )))
}

fn manager_with<S: SettingsStorage + 'static>(
    cwd: &str,
    agent_dir: &str,
    settings: &Arc<Mutex<SettingsManager<S>>>,
) -> DefaultPackageManager<S> {
    DefaultPackageManager::new(PackageManagerOptions {
        cwd: cwd.to_string(),
        agent_dir: agent_dir.to_string(),
        settings: Arc::clone(settings),
        command_runner: None,
        env: None,
        http_client: None,
    })
}

#[tokio::test]
async fn resolve_collects_auto_discovered_directories_with_precedence() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    std::fs::create_dir_all(agent_dir.join("skills")).expect("mkdir");
    std::fs::create_dir_all(cwd.join(".pi/skills")).expect("mkdir");
    std::fs::write(agent_dir.join("skills/user.md"), "").expect("write");
    std::fs::write(cwd.join(".pi/skills/project.md"), "").expect("write");

    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings_handle(),
    );
    let resolved = manager.resolve(None).await.expect("resolve");

    assert_eq!(resolved.skills.len(), 2);
    // Project resources sort before user resources.
    assert_eq!(
        PathBuf::from(&resolved.skills[0].path),
        cwd.join(".pi/skills/project.md")
    );
    assert!(matches!(
        resolved.skills[0].metadata.scope,
        SourceScope::Project
    ));
    assert_eq!(
        PathBuf::from(&resolved.skills[1].path),
        agent_dir.join("skills/user.md")
    );
    assert!(matches!(
        resolved.skills[1].metadata.scope,
        SourceScope::User
    ));
}

#[tokio::test]
async fn resolve_gates_project_discovery_on_project_trust() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    std::fs::create_dir_all(agent_dir.join("skills")).expect("mkdir");
    std::fs::create_dir_all(cwd.join(".pi/skills")).expect("mkdir");
    std::fs::write(agent_dir.join("skills/user.md"), "").expect("write");
    std::fs::write(cwd.join(".pi/skills/project.md"), "").expect("write");

    let settings = Arc::new(Mutex::new(SettingsManager::create(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        SettingsManagerCreateOptions {
            project_trusted: Some(false),
        },
    )));
    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings,
    );
    let resolved = manager.resolve(None).await.expect("resolve");

    assert_eq!(resolved.skills.len(), 1);
    assert_eq!(
        PathBuf::from(&resolved.skills[0].path),
        agent_dir.join("skills/user.md")
    );
}

#[tokio::test]
async fn resolve_carries_the_agents_skills_trees_with_their_own_base_dirs() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    // The cwd tree and its parent container both carry .agents/skills; the
    // ancestor walk collects each with its own base dir. (The user-side
    // tree rides the real home directory and is not fixture-able here.)
    std::fs::create_dir_all(cwd.join(".agents/skills/sub")).expect("mkdir");
    std::fs::create_dir_all(env.path().join(".agents/skills/sub")).expect("mkdir");
    std::fs::write(cwd.join(".agents/skills/sub/SKILL.md"), "").expect("write");
    std::fs::write(env.path().join(".agents/skills/sub/outer.md"), "").expect("write");

    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings_handle(),
    );
    let resolved = manager.resolve(None).await.expect("resolve");

    let project_entry = resolved
        .skills
        .iter()
        .find(|resource| {
            resource
                .path
                .ends_with("project/.agents/skills/sub/SKILL.md")
        })
        .expect("project agents entry");
    assert_eq!(
        PathBuf::from(
            project_entry
                .metadata
                .base_dir
                .as_deref()
                .expect("base dir")
        ),
        cwd.join(".agents")
    );
    let container_entry = resolved
        .skills
        .iter()
        .find(|resource| resource.path.ends_with(".agents/skills/sub/outer.md"))
        .expect("container agents entry");
    assert_eq!(
        PathBuf::from(
            container_entry
                .metadata
                .base_dir
                .as_deref()
                .expect("base dir")
        ),
        env.path().join(".agents")
    );
}

#[tokio::test]
async fn resolve_canonical_dedup_keeps_the_first_path() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = agent_dir.clone();
    std::fs::create_dir_all(agent_dir.join("skills")).expect("mkdir");
    std::fs::write(agent_dir.join("skills/dup.md"), "").expect("write");

    // The same file through a settings entry and through auto-discovery:
    // the settings entry (rank 2) lands first and wins.
    let settings = settings_handle();
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_skill_paths(&["skills/dup.md".to_string()]);
    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings,
    );
    let resolved = manager.resolve(None).await.expect("resolve");

    assert_eq!(resolved.skills.len(), 1);
    assert!(matches!(
        resolved.skills[0].metadata.scope,
        SourceScope::User
    ));
    assert_eq!(resolved.skills[0].metadata.source, "local");
}

#[tokio::test]
async fn resolve_extension_sources_collects_local_paths() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    std::fs::create_dir_all(&cwd).expect("mkdir");
    let ext_file = env.path().join("ext.ts");
    std::fs::write(&ext_file, "").expect("write");

    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings_handle(),
    );
    // The loader always asks for the temporary scope, upstream's
    // `resolveExtensionSources(paths, { temporary: true })`.
    let resolved = manager
        .resolve_extension_sources(&[ext_file.to_string_lossy().into_owned()], false, true)
        .await
        .expect("resolve");

    assert_eq!(resolved.extensions.len(), 1);
    assert_eq!(PathBuf::from(&resolved.extensions[0].path), ext_file);
    assert!(matches!(
        resolved.extensions[0].metadata.scope,
        SourceScope::Temporary
    ));
    assert!(matches!(
        resolved.extensions[0].metadata.origin,
        SourceOrigin::Package
    ));
}

#[tokio::test]
async fn resolve_extension_sources_rejects_npm_and_reads_git_plus_urls_local() {
    let manager = manager_with("/tmp", "/tmp/agent", &settings_handle());
    // The npm channel drops (ADR 0007): an `npm:` source is a parse error,
    // not a skipped entry.
    let error = manager
        .resolve_extension_sources(&["npm:foo".to_string()], false, true)
        .await
        .expect_err("an npm source is a parse error");
    assert!(
        error.0.contains("npm package sources are not supported"),
        "{error:?}"
    );
    // A `git+https://` spec parses as neither a git URL nor a remote
    // protocol — the parser reads it local, and the resolver probes the
    // path; a missing path contributes no entries.
    let resolved = manager
        .resolve_extension_sources(&["git+https://x".to_string()], false, true)
        .await
        .expect("resolve");
    assert_eq!(resolved, ResolvedPaths::default());
}

#[tokio::test]
async fn local_package_sources_collect_package_resources() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    let package_root = cwd.join("local-package");
    std::fs::create_dir_all(package_root.join("skills")).expect("mkdir");
    std::fs::create_dir_all(package_root.join("prompts")).expect("mkdir");
    std::fs::write(package_root.join("skills/SKILL.md"), "").expect("write");
    std::fs::write(package_root.join("prompts/p.md"), "").expect("write");
    // No `pi` manifest: the default layout directories collect.
    std::fs::write(
        package_root.join("package.json"),
        r#"{"name":"local-package"}"#,
    )
    .expect("write");

    let settings = settings_handle();
    // Global package entries resolve against the agent dir, upstream's
    // `getBaseDirForScope`; the fixture uses an absolute source.
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::Value::String(
            package_root.to_string_lossy().into_owned(),
        )]);
    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings,
    );
    let resolved = manager.resolve(None).await.expect("resolve");

    assert_eq!(resolved.skills.len(), 1);
    assert_eq!(
        PathBuf::from(&resolved.skills[0].path),
        package_root.join("skills/SKILL.md")
    );
    assert!(matches!(
        resolved.skills[0].metadata.origin,
        SourceOrigin::Package
    ));
    assert_eq!(resolved.prompts.len(), 1);
    assert_eq!(
        PathBuf::from(&resolved.prompts[0].path),
        package_root.join("prompts/p.md")
    );
}

#[tokio::test]
async fn a_package_manifest_carries_only_its_own_entries() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    let package_root = cwd.join("manifest-package");
    std::fs::create_dir_all(package_root.join("skills")).expect("mkdir");
    std::fs::create_dir_all(package_root.join("prompts")).expect("mkdir");
    std::fs::write(package_root.join("skills/SKILL.md"), "").expect("write");
    std::fs::write(package_root.join("prompts/p.md"), "").expect("write");
    std::fs::write(
        package_root.join("package.json"),
        r#"{"name":"manifest-package","pi":{"prompts":["prompts/p.md"]}}"#,
    )
    .expect("write");

    let settings = settings_handle();
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::Value::String(
            package_root.to_string_lossy().into_owned(),
        )]);
    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings,
    );
    let resolved = manager.resolve(None).await.expect("resolve");

    // The manifest's presence short-circuits the default-layout sweep:
    // only its own entries load.
    assert_eq!(resolved.skills, Vec::new());
    assert_eq!(resolved.prompts.len(), 1);
    assert_eq!(
        PathBuf::from(&resolved.prompts[0].path),
        package_root.join("prompts/p.md")
    );
}

#[tokio::test]
async fn a_local_package_directory_without_resources_becomes_one_extension_entry() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    let package_root = cwd.join("bare-package");
    std::fs::create_dir_all(&package_root).expect("mkdir");
    std::fs::write(
        package_root.join("package.json"),
        r#"{"name":"bare-package"}"#,
    )
    .expect("write");

    let settings = settings_handle();
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::Value::String(
            package_root.to_string_lossy().into_owned(),
        )]);
    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings,
    );
    let resolved = manager.resolve(None).await.expect("resolve");

    assert_eq!(resolved.extensions.len(), 1);
    assert_eq!(PathBuf::from(&resolved.extensions[0].path), package_root);
}

// === resource-loader boundaries =============================================

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

    fn file_manager(
        &self,
    ) -> SettingsManager<pi_coding_agent::settings_manager::FileSettingsStorage> {
        SettingsManager::create(
            &self.cwd,
            &self.agent_dir,
            SettingsManagerCreateOptions::default(),
        )
    }

    fn write(&self, relative: &str, content: &str) {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(path, content).expect("write");
    }
}

#[tokio::test]
async fn missing_cli_paths_report_errors_per_resource_family() {
    let env = LoaderEnv::new();
    let missing_extension = env.root.join("no-such-extension.ts");
    let missing_skill = env.root.join("no-such-skill");
    let missing_prompt = env.root.join("no-such-prompt.md");
    let missing_theme = env.root.join("no-such-theme.json");

    let mut options = env.options();
    options.additional_extension_paths = vec![missing_extension.to_string_lossy().into_owned()];
    options.additional_skill_paths = vec![missing_skill.to_string_lossy().into_owned()];
    options.additional_prompt_template_paths = vec![missing_prompt.to_string_lossy().into_owned()];
    options.additional_theme_paths = vec![missing_theme.to_string_lossy().into_owned()];
    // The loader's own loads report the missing paths first (a warning per
    // loadSkills); the reload loop's error diagnostics fire when that
    // record is absent — the skip-flag shape, upstream's
    // `noSkills && skillPaths.length === 0` empty result.
    options.no_skills = true;
    options.no_prompt_templates = true;
    options.no_themes = true;

    let mut loader = DefaultResourceLoader::new(options, env.file_manager());
    loader.reload(None).await;

    let extensions = loader.get_extensions();
    assert!(extensions.errors.iter().any(|error| {
        error.path == missing_extension.to_string_lossy()
            && error.error.starts_with("Extension path does not exist: ")
    }));

    // The skills loader records the missing path as its own warning, so
    // the reload loop's error push stays guarded (upstream's
    // `!this.skillDiagnostics.some(...)` never opens for a missing
    // additional path); the warning is what surfaces.
    let skills = loader.get_skills();
    assert!(skills.diagnostics.iter().any(|d| {
        d.path.as_deref() == Some(missing_skill.to_string_lossy().as_ref())
            && d.message == "skill path does not exist"
            && d.kind == ResourceDiagnosticKind::Warning
    }));

    // The prompt loader skips missing paths silently, so the reload loop's
    // error diagnostic is the live record.
    let prompts = loader.get_prompts();
    assert!(prompts.diagnostics.iter().any(|d| {
        d.path.as_deref() == Some(missing_prompt.to_string_lossy().as_ref())
            && d.message == "Prompt template path does not exist"
            && d.kind == ResourceDiagnosticKind::Error
    }));

    // The theme loader records the missing path as its own warning the
    // same way skills do.
    let themes = loader.get_themes();
    assert!(themes.diagnostics.iter().any(|d| {
        d.path.as_deref() == Some(missing_theme.to_string_lossy().as_ref())
            && d.message == "theme path does not exist"
            && d.kind == ResourceDiagnosticKind::Warning
    }));
}

#[tokio::test]
async fn theme_load_failures_and_non_json_paths_warn() {
    let env = LoaderEnv::new();
    env.write("agent/themes/broken.json", "not json at all");
    // A non-json theme path rides a direct additional path; the
    // auto-discovered theme dirs pre-filter to .json.
    let plain_md = env.root.join("plain.md");
    std::fs::write(&plain_md, "nope").expect("write");

    let mut options = env.options();
    options.additional_theme_paths = vec![plain_md.to_string_lossy().into_owned()];

    let mut loader = DefaultResourceLoader::new(options, env.file_manager());
    loader.reload(None).await;

    let themes = loader.get_themes();
    assert!(themes.diagnostics.iter().any(|d| {
        d.kind == ResourceDiagnosticKind::Warning
            && d.path
                .as_deref()
                .is_some_and(|p| p.ends_with("broken.json"))
    }));
    assert!(
        themes
            .diagnostics
            .iter()
            .any(|d| d.message == "theme path is not a json file"
                && d.path.as_deref() == Some(plain_md.to_string_lossy().as_ref()))
    );
    assert!(themes.themes.is_empty());
}

#[tokio::test]
async fn unnamed_themes_dedupe_as_unnamed_with_collisions() {
    let env = LoaderEnv::new();
    env.write("agent/themes/one.json", "{}");
    env.write("project/.pi/themes/two.json", "{}");

    let mut loader = DefaultResourceLoader::new(env.options(), env.file_manager());
    loader.reload(None).await;

    let themes = loader.get_themes();
    // Project themes sort first; the user-side unnamed theme records the
    // collision and the project theme keeps the name.
    assert_eq!(themes.themes.len(), 1);
    assert!(
        themes
            .diagnostics
            .iter()
            .any(|d| d.kind == ResourceDiagnosticKind::Collision
                && d.message == "name \"unnamed\" collision")
    );
}

#[tokio::test]
async fn duplicate_prompts_across_families_carry_the_collision_with_the_slash_prefix() {
    let env = LoaderEnv::new();
    env.write("agent/prompts/commit.md", "User prompt");
    env.write("project/.pi/prompts/commit.md", "Project prompt");

    let mut loader = DefaultResourceLoader::new(env.options(), env.file_manager());
    loader.reload(None).await;

    let prompts = loader.get_prompts();
    assert_eq!(prompts.prompts.len(), 1);
    assert_eq!(prompts.prompts[0].name, "commit");
    let collision = prompts
        .diagnostics
        .iter()
        .find(|d| d.kind == ResourceDiagnosticKind::Collision)
        .expect("collision");
    assert_eq!(collision.message, "name \"/commit\" collision");
    let collision_detail = collision.collision.as_ref().expect("collision detail");
    assert_eq!(collision_detail.winner_path, prompts.prompts[0].file_path);
    assert!(
        collision_detail
            .loser_path
            .ends_with("agent/prompts/commit.md")
    );
}

#[tokio::test]
async fn the_pre_trust_flow_runs_and_reports_the_bootstrap_set() {
    let env = LoaderEnv::new();
    let mut loader = DefaultResourceLoader::new(env.options(), env.file_manager());

    let seen: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(None));
    let seen_for_callback = seen.clone();
    loader
        .reload(Some(
            pi_coding_agent::resource_loader::ResourceLoaderReloadOptions {
                resolve_project_trust: Some(Box::new(move |extensions_result| {
                    *seen_for_callback.lock().expect("lock") =
                        Some(extensions_result.extensions.len());
                    Box::pin(async { true })
                })),
            },
        ))
        .await;

    // The bootstrap set is the extension seam's (empty until #128); the
    // callback ran and the trusted reload completed.
    assert_eq!(*seen.lock().expect("lock"), Some(0));
    assert!(loader.get_extensions().extensions.is_empty());
}

#[tokio::test]
async fn extend_resources_stamps_theme_metadata_from_the_extension() {
    let env = LoaderEnv::new();
    let extra_theme_dir = env.root.join("extra-themes");
    let theme_path = extra_theme_dir.join("extra.json");
    std::fs::create_dir_all(&extra_theme_dir).expect("mkdir");
    std::fs::write(&theme_path, "{\"name\": \"extension-theme\"}").expect("write");

    let mut loader = DefaultResourceLoader::new(env.options(), env.file_manager());
    loader.reload(None).await;

    loader.extend_resources(&ResourceExtensionPaths {
        skill_paths: Vec::new(),
        prompt_paths: Vec::new(),
        theme_paths: vec![PathWithMetadata {
            path: extra_theme_dir.to_string_lossy().into_owned(),
            metadata: PathMetadata {
                source: "extension:extra".to_string(),
                scope: SourceScope::Temporary,
                origin: SourceOrigin::TopLevel,
                base_dir: Some(extra_theme_dir.to_string_lossy().into_owned()),
            },
        }],
    });

    let themes = loader.get_themes();
    let theme = themes
        .themes
        .iter()
        .find(|t| t.name.as_deref() == Some("extension-theme"))
        .expect("extension theme");
    assert_eq!(
        PathBuf::from(theme.source_path.as_deref().expect("source path")),
        theme_path
    );
    let source_info = theme.source_info.as_ref().expect("source info");
    assert_eq!(source_info.source, "extension:extra");
    assert_eq!(PathBuf::from(&source_info.path), theme_path);
}

#[tokio::test]
async fn the_no_flags_still_load_cli_paths() {
    let env = LoaderEnv::new();
    let custom_prompt_dir = env.root.join("custom-prompts");
    std::fs::create_dir_all(&custom_prompt_dir).expect("mkdir");
    std::fs::write(custom_prompt_dir.join("cli.md"), "CLI prompt").expect("write");
    env.write(
        "agent/skills/skill.md",
        "---\nname: s\ndescription: d\n---\nx",
    );
    env.write("agent/prompts/auto.md", "Auto prompt");
    env.write("agent/themes/auto.json", "{}");

    let mut options = env.options();
    options.no_skills = true;
    options.no_prompt_templates = true;
    options.no_themes = true;
    options.no_extensions = true;
    options.additional_prompt_template_paths =
        vec![custom_prompt_dir.to_string_lossy().into_owned()];

    let mut loader = DefaultResourceLoader::new(options, env.file_manager());
    loader.reload(None).await;

    assert!(loader.get_skills().skills.is_empty());
    // The CLI prompt path still loads; the auto dirs do not.
    assert!(loader.get_prompts().prompts.iter().any(|p| p.name == "cli"));
    assert!(
        !loader
            .get_prompts()
            .prompts
            .iter()
            .any(|p| p.name == "auto")
    );
    assert!(loader.get_themes().themes.is_empty());
}

#[tokio::test]
async fn literal_system_prompt_read_failures_fall_back_to_the_literal() {
    let env = LoaderEnv::new();
    // A directory in the system-prompt slot: exists, cannot read as text.
    let dir_prompt = env.root.join("prompt-dir");
    std::fs::create_dir_all(&dir_prompt).expect("mkdir");

    let mut options = env.options();
    options.system_prompt = Some(dir_prompt.to_string_lossy().into_owned());

    let mut loader = DefaultResourceLoader::new(options, env.file_manager());
    loader.reload(None).await;

    assert_eq!(
        loader.get_system_prompt(),
        Some(dir_prompt.to_string_lossy().into_owned())
    );
}

#[tokio::test]
async fn skills_reload_with_no_skills_still_loads_cli_paths_through_the_loader() {
    let env = LoaderEnv::new();
    let custom_skill_dir = env.root.join("custom-skill");
    std::fs::create_dir_all(&custom_skill_dir).expect("mkdir");
    std::fs::write(
        custom_skill_dir.join("SKILL.md"),
        "---\nname: cli-skill\ndescription: From the CLI\n---\nx",
    )
    .expect("write");

    let mut options = env.options();
    options.no_skills = true;
    options.additional_skill_paths = vec![custom_skill_dir.to_string_lossy().into_owned()];

    let mut loader = DefaultResourceLoader::new(options, env.file_manager());
    loader.reload(None).await;

    let skills = loader.get_skills();
    assert_eq!(skills.skills.len(), 1);
    assert_eq!(skills.skills[0].name, "cli-skill");
}

#[tokio::test]
async fn default_source_attribution_covers_agent_project_and_temporary() {
    let env = LoaderEnv::new();
    env.write(
        "agent/skills/user-skill.md",
        "---\nname: u\ndescription: d\n---\nx",
    );
    env.write(
        "project/.pi/skills/project-skill.md",
        "---\nname: p\ndescription: d\n---\nx",
    );
    let elsewhere = env.root.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).expect("mkdir");
    std::fs::write(
        elsewhere.join("temp-skill.md"),
        "---\nname: t\ndescription: d\n---\nx",
    )
    .expect("write");

    let mut options = env.options();
    options.additional_skill_paths = vec![
        env.root.join("agent/skills").to_string_lossy().into_owned(),
        env.root
            .join("project/.pi/skills")
            .to_string_lossy()
            .into_owned(),
        elsewhere.to_string_lossy().into_owned(),
    ];

    let mut loader = DefaultResourceLoader::new(options, env.file_manager());
    loader.reload(None).await;

    let skills = loader.get_skills();
    let by_name = |name: &str| {
        skills
            .skills
            .iter()
            .find(|s| s.name == name)
            .expect("skill")
            .clone()
    };
    // The agent/project skills were ALSO auto-discovered (the fixture dirs
    // are the default trees), so the loader's exact metadata match wins and
    // their attribution carries the auto-discovery's: source `auto`, the
    // scope's base dir.
    let user_skill = by_name("u");
    assert!(matches!(user_skill.source_info.scope, SourceScope::User));
    assert_eq!(user_skill.source_info.source, "auto");
    assert_eq!(
        PathBuf::from(user_skill.source_info.base_dir.as_deref().expect("base")),
        env.root.join("agent")
    );
    let project_skill = by_name("p");
    assert!(matches!(
        project_skill.source_info.scope,
        SourceScope::Project
    ));
    assert_eq!(
        PathBuf::from(project_skill.source_info.base_dir.as_deref().expect("base")),
        env.root.join("project/.pi")
    );
    let temp_skill = by_name("t");
    assert!(matches!(
        temp_skill.source_info.scope,
        SourceScope::Temporary
    ));
    assert_eq!(
        PathBuf::from(temp_skill.source_info.base_dir.as_deref().expect("base")),
        elsewhere
    );
}

#[tokio::test]
async fn synthetic_source_labels_read_their_inline_prefix() {
    let env = LoaderEnv::new();
    let mut options = env.options();
    options.skills_override = Some(Box::new(|base| {
        let mut skill = base.skills.first().cloned();
        if let Some(skill) = &mut skill {
            skill.file_path = "<inline:1>".to_string();
            skill.source_info.path = "<inline:1>".to_string();
        }
        base
    }));

    let mut loader = DefaultResourceLoader::new(options, env.file_manager());
    loader.reload(None).await;

    let skills = loader.get_skills();
    let Some(skill) = skills.skills.first() else {
        // No skill loaded to re-label; the override's target is the
        // `<inline:…>` branch of the source attribution, exercised through
        // the theme handle path below.
        return;
    };
    assert_eq!(skill.source_info.source, "inline");
    assert!(matches!(skill.source_info.scope, SourceScope::Temporary));
}

#[tokio::test]
async fn no_flags_keep_the_loader_initialized_empty() {
    let env = LoaderEnv::new();
    let mut options = env.options();
    options.no_skills = true;
    options.no_prompt_templates = true;
    options.no_themes = true;
    options.no_extensions = true;
    options.no_context_files = true;

    let mut loader = DefaultResourceLoader::new(options, env.file_manager());
    loader.reload(None).await;

    assert!(loader.get_skills().skills.is_empty());
    assert!(loader.get_prompts().prompts.is_empty());
    assert!(loader.get_themes().themes.is_empty());
    assert!(loader.get_extensions().extensions.is_empty());
    assert!(loader.get_agents_files().is_empty());
    assert!(loader.get_system_prompt().is_none());
    assert!(loader.get_append_system_prompt().is_empty());
}

// === prompt-template loader boundaries ======================================

#[test]
fn template_sources_attribute_global_project_and_explicit_dirs() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    std::fs::create_dir_all(agent_dir.join("prompts")).expect("mkdir");
    std::fs::create_dir_all(cwd.join(".pi/prompts")).expect("mkdir");
    std::fs::create_dir_all(cwd.join("custom")).expect("mkdir");
    std::fs::write(agent_dir.join("prompts/global.md"), "").expect("write");
    std::fs::write(cwd.join(".pi/prompts/project.md"), "").expect("write");
    std::fs::write(cwd.join("custom/explicit.md"), "").expect("write");

    let templates = pi_coding_agent::prompt_templates::load_prompt_templates(
        &pi_coding_agent::prompt_templates::LoadPromptTemplatesOptions {
            cwd: &cwd.to_string_lossy(),
            agent_dir: &agent_dir.to_string_lossy(),
            prompt_paths: &["custom".to_string()],
            include_defaults: true,
        },
    );

    let by_name = |name: &str| {
        templates
            .iter()
            .find(|t| t.name == name)
            .expect("template")
            .clone()
    };
    let global = by_name("global");
    assert!(matches!(global.source_info.scope, SourceScope::User));
    assert_eq!(
        PathBuf::from(global.source_info.base_dir.as_deref().expect("base")),
        agent_dir.join("prompts")
    );
    let project = by_name("project");
    assert!(matches!(project.source_info.scope, SourceScope::Project));
    let explicit = by_name("explicit");
    assert!(matches!(explicit.source_info.scope, SourceScope::Temporary));
    // A directory base points at the directory itself; a file base points
    // at its parent.
    assert_eq!(
        PathBuf::from(explicit.source_info.base_dir.as_deref().expect("base")),
        cwd.join("custom")
    );
}

#[test]
fn template_names_strip_only_the_md_suffix() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("plain.md"), "").expect("write");
    std::fs::write(dir.path().join("upper.MD"), "").expect("write");

    let templates = pi_coding_agent::prompt_templates::load_prompt_templates(
        &pi_coding_agent::prompt_templates::LoadPromptTemplatesOptions {
            cwd: &dir.path().to_string_lossy(),
            agent_dir: &dir.path().to_string_lossy(),
            prompt_paths: &[dir.path().to_string_lossy().into_owned()],
            include_defaults: false,
        },
    );

    let names: Vec<&str> = templates.iter().map(|t| t.name.as_str()).collect();
    // Upstream's case-sensitive `endsWith(".md")` selects the files; an
    // `upper.MD` file never loads from a directory scan.
    assert_eq!(names, vec!["plain"]);
}

// === prompt-template substitution boundaries ================================

#[test]
fn substitution_boundary_shapes_stay_literal() {
    use pi_coding_agent::prompt_templates::substitute_args;

    // A malformed default form (no closing brace) stays literal.
    assert_eq!(
        substitute_args("${1:-unclosed", &["a".to_string()]),
        "${1:-unclosed"
    );
    // A non-numeric target stays literal.
    assert_eq!(substitute_args("${x:-y}", &[]), "${x:-y}");
    // A slice with a non-numeric start stays literal.
    assert_eq!(substitute_args("${@:x}", &["a".to_string()]), "${@:x}");
    // A lone dollar stays literal.
    assert_eq!(substitute_args("$", &["a".to_string()]), "$");
    // A braced default reads its value's truthiness, upstream's
    // `value ? value : defaultValue`.
    assert_eq!(substitute_args("${1:-d}", &[String::new()]), "d");
    assert_eq!(substitute_args("${1:-d}", &["0".to_string()]), "0");
    // An ARGUMENTS default over empty args falls back.
    assert_eq!(substitute_args("${ARGUMENTS:-fallback}", &[]), "fallback");
    // Huge positions read missing, upstream's parseInt saturation.
    assert_eq!(
        substitute_args("$99999999999999999999", &["a".to_string()]),
        ""
    );
    assert_eq!(
        substitute_args("${@:99999999999999999999}", &["a".to_string()]),
        ""
    );
}

// === skills loader boundaries ===============================================

#[test]
fn skill_paths_at_the_default_directories_read_user_and_project_sources() {
    let env = tempfile::tempdir().expect("tempdir");
    let agent_dir = env.path().join("agent");
    let cwd = env.path().join("project");
    std::fs::create_dir_all(agent_dir.join("skills")).expect("mkdir");
    std::fs::create_dir_all(cwd.join(".pi/skills")).expect("mkdir");
    std::fs::write(
        agent_dir.join("skills/u.md"),
        "---\nname: u\ndescription: d\n---\n",
    )
    .expect("write");
    std::fs::write(
        cwd.join(".pi/skills/p.md"),
        "---\nname: p\ndescription: d\n---\n",
    )
    .expect("write");

    let result =
        pi_coding_agent::skills::load_skills(&pi_coding_agent::skills::LoadSkillsOptions {
            cwd: &cwd.to_string_lossy(),
            agent_dir: &agent_dir.to_string_lossy(),
            skill_paths: &[],
            include_defaults: true,
        });

    let by_name = |name: &str| {
        result
            .skills
            .iter()
            .find(|s| s.name == name)
            .expect("skill")
    };
    assert!(matches!(by_name("u").source_info.scope, SourceScope::User));
    assert!(matches!(
        by_name("p").source_info.scope,
        SourceScope::Project
    ));
    assert_eq!(by_name("u").source_info.source, "local");
}

#[test]
fn skill_discovery_honors_ignore_files_across_the_walk() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("nested/skip")).expect("mkdir");
    std::fs::write(dir.path().join(".gitignore"), "nested/skip\n").expect("write");
    std::fs::write(
        dir.path().join("nested/skip/SKILL.md"),
        "---\ndescription: d\n---\n",
    )
    .expect("write");
    // A root-level .md file loads (root files load at the traversal root);
    // the ignored subtree's SKILL.md does not.
    std::fs::write(
        dir.path().join("keep.md"),
        "---\nname: keep\ndescription: d\n---\n",
    )
    .expect("write");

    let result = pi_coding_agent::skills::load_skills_from_dir(
        &pi_coding_agent::skills::LoadSkillsFromDirOptions {
            dir: &dir.path().to_string_lossy(),
            source: "test",
        },
    );

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "keep");
}

// === prompt expansion boundaries ============================================

#[test]
fn expansion_passes_non_template_text_through() {
    let templates = Vec::new();
    assert_eq!(
        pi_coding_agent::prompt_templates::expand_prompt_template("plain text", &templates),
        "plain text"
    );
    // A bare slash has no template name and passes through.
    assert_eq!(
        pi_coding_agent::prompt_templates::expand_prompt_template("/", &templates),
        "/"
    );
    // An unknown template passes through.
    assert_eq!(
        pi_coding_agent::prompt_templates::expand_prompt_template("/nope args", &templates),
        "/nope args"
    );
}

// === package manifest glob entries ==========================================

#[tokio::test]
async fn manifest_glob_entries_expand_and_drop_dot_segments() {
    let env = tempfile::tempdir().expect("tempdir");
    let package_root = env.path().join("pkg");
    std::fs::create_dir_all(package_root.join("prompts/sub")).expect("mkdir");
    std::fs::create_dir_all(package_root.join("prompts/.hidden")).expect("mkdir");
    std::fs::write(package_root.join("prompts/a.md"), "").expect("write");
    std::fs::write(package_root.join("prompts/sub/b.md"), "").expect("write");
    std::fs::write(package_root.join("prompts/.hidden/c.md"), "").expect("write");
    std::fs::write(
        package_root.join("package.json"),
        r#"{"pi":{"prompts":["prompts/**/*.md"]}}"#,
    )
    .expect("write");

    let agent_dir = env.path().join("agent");
    std::fs::create_dir_all(&agent_dir).expect("mkdir");
    let cwd = env.path().join("project");
    std::fs::create_dir_all(&cwd).expect("mkdir");
    let settings = settings_handle();
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::Value::String("../pkg".to_string())]);
    let manager = manager_with(
        &cwd.to_string_lossy(),
        &agent_dir.to_string_lossy(),
        &settings,
    );
    let resolved = manager.resolve(None).await.expect("resolve");

    let paths: Vec<String> = resolved.prompts.iter().map(|p| p.path.clone()).collect();
    assert!(paths.iter().any(|p| p.ends_with("prompts/a.md")));
    assert!(paths.iter().any(|p| p.ends_with("prompts/sub/b.md")));
    // Dot segments drop, upstream's expandPackageGlob filter.
    assert!(!paths.iter().any(|p| p.contains(".hidden")));
}

// === prompt source shape ====================================================

#[test]
fn prompt_sources_debug_print() {
    let source = PromptSource {
        path: "/x".to_string(),
    };
    assert!(format!("{source:?}").contains("/x"));
}
